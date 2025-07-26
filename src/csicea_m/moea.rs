// Community Structure Identification Co‑evolutionary Algorithm

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::distr::weighted::WeightedIndex;
use rand::{prelude::*, rng};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub nodes: Vec<usize>,
    pub obj1: f64,  // Conductance (minimize)
    pub obj2: f64,  // Key nodes count (minimize)
}

impl Chromosome {
    pub fn new(nodes: Vec<usize>, obj1: f64, obj2: f64) -> Self {
        Self {
            nodes,
            obj1,
            obj2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Community {
    pub nodes: Vec<usize>,
}

pub struct MOEA {
    adj_matrix: Array2<f64>,
    num_nodes: usize,
    popsize: usize,
    niche: usize,
    max_gen: usize,
    cross_rate: f64,
    mutate_rate: f64,
    alpha: f64,
    degree: Array1<f64>,
    edge_num: f64,
    avg_degree: f64,
    active_nodes: Vec<usize>,
    key_nodes: Option<Vec<usize>>,
    // Cache for expensive computations
    neighbor_cache: FxHashMap<usize, Vec<usize>>,
    verbose: bool,
}

impl MOEA {
    pub fn new(
        adj_matrix: Array2<f64>,
        popsize: usize,
        niche: usize,
        max_gen: usize,
        cross_rate: f64,
        mutate_rate: f64,
        alpha: f64,
        verbose: bool,
    ) -> Self {
        let num_nodes = adj_matrix.nrows();

        // Calculate network properties
        let degree: Array1<f64> = adj_matrix.sum_axis(Axis(1));
        let edge_num: f64 = degree.sum() / 2.0;
        let avg_degree: f64 = degree.mean().unwrap_or(0.0);

        // Remove isolated nodes
        let active_nodes: Vec<usize> = degree
            .iter()
            .enumerate()
            .filter(|(_, d)| **d > 0.0)
            .map(|(i, _)| i)
            .collect();

        let filtered_adj_matrix = if active_nodes.len() < num_nodes {
            let indices: Vec<_> = active_nodes.iter().cloned().collect();
            let mut new_matrix = Array2::zeros((active_nodes.len(), active_nodes.len()));
            for (i, &row_idx) in indices.iter().enumerate() {
                for (j, &col_idx) in indices.iter().enumerate() {
                    new_matrix[[i, j]] = adj_matrix[[row_idx, col_idx]];
                }
            }
            new_matrix
        } else {
            adj_matrix
        };

        let filtered_degree: Array1<f64> = active_nodes.iter().map(|&i| degree[i]).collect();

        // Pre-compute neighbor cache for all nodes using FxHashMap
        let mut neighbor_cache = FxHashMap::default();
        for i in 0..active_nodes.len() {
            let neighbors: Vec<usize> = (0..active_nodes.len())
                .filter(|&j| filtered_adj_matrix[[i, j]] > 0.0)
                .collect();
            neighbor_cache.insert(i, neighbors);
        }

        Self {
            adj_matrix: filtered_adj_matrix,
            num_nodes: active_nodes.len(),
            popsize,
            niche,
            max_gen,
            cross_rate,
            mutate_rate,
            alpha,
            degree: filtered_degree,
            edge_num,
            avg_degree,
            active_nodes,
            key_nodes: None,
            neighbor_cache,
            verbose
        }
    }

    /// Get cached neighbors for a node
    #[inline]
    fn get_neighbors(&self, node: usize) -> &Vec<usize> {
        &self.neighbor_cache[&node]
    }

    /// Find key nodes with improved algorithm
    pub fn find_key_nodes(&mut self) -> Vec<usize> {
        let mut key_nodes = Vec::new();

        // Find all nodes with degree larger than average degree using FxHashSet
        let mut lset: FxHashSet<usize> = self
            .degree
            .indexed_iter()
            .filter(|(_, d)| **d > self.avg_degree)
            .map(|(i, _)| i)
            .collect();

        if lset.is_empty() {
            // If no nodes above average, take top 10% by degree
            let top_count = (self.num_nodes / 10).max(1);
            let mut degree_indices: Vec<_> = self.degree.indexed_iter().collect();
            degree_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            lset = degree_indices
                .into_iter()
                .take(top_count)
                .map(|(i, _)| i)
                .collect();
        }

        while !lset.is_empty() {
            // Find node with maximum degree in Lset
            let max_node = *lset
                .iter()
                .max_by(|&&a, &&b| self.degree[a].partial_cmp(&self.degree[b]).unwrap())
                .unwrap();

            key_nodes.push(max_node);

            // Use cached neighbors and FxHashSet
            let neighbors: FxHashSet<usize> = self
                .get_neighbors(max_node)
                .iter()
                .filter(|&n| lset.contains(n))
                .cloned()
                .collect();

            // Remove max_node and all its neighbors from Lset
            lset.remove(&max_node);
            for neighbor in neighbors {
                lset.remove(&neighbor);
            }
        }

        self.key_nodes = Some(key_nodes.clone());
        key_nodes
    }

    #[inline]
    pub fn calculate_conductance(&self, community_nodes: &[usize]) -> f64 {
        if community_nodes.is_empty() {
            return 1.0;
        }

        let community_set: FxHashSet<usize> = community_nodes.iter().cloned().collect();

        let mut e_ci_vci = 0.0; // E(Ci, V\Ci) - edges between community and external nodes
        let mut e_ci_v = 0.0; // E(Ci, V) - total degree of community nodes

        // Calculate E(Ci, V\Ci) and E(Ci, V) in one pass
        for &node in community_nodes {
            let neighbors = self.get_neighbors(node);
            e_ci_v += neighbors.len() as f64; // Use cached degree from neighbors

            // Count edges to external nodes using cached neighbors
            for &neighbor in neighbors {
                if !community_set.contains(&neighbor) {
                    e_ci_vci += self.adj_matrix[[node, neighbor]];
                }
            }
        }

        // Calculate E(V\Ci, V) = total_edges - E(Ci, V) + E(Ci, V\Ci)
        // Since total degree = 2 * edge_num, and we want degree of complement
        let e_vci_v = (2.0 * self.edge_num) - e_ci_v + e_ci_vci;

        let denominator = e_ci_v.min(e_vci_v);

        if denominator == 0.0 {
            1.0
        } else {
            e_ci_vci / denominator
        }
    }

    /// Count number of key nodes in community (unchanged but inlined)
    #[inline]
    pub fn count_key_nodes_in_community(&self, community_nodes: &[usize]) -> f64 {
        if let Some(ref key_nodes) = self.key_nodes {
            let community_set: FxHashSet<usize> = community_nodes.iter().cloned().collect();
            key_nodes
                .iter()
                .filter(|&n| community_set.contains(n))
                .count() as f64
        } else {
            0.0
        }
    }

    /// Initialize weight vectors - optimized with pre-allocation
    pub fn init_weights(&self) -> (Array2<f64>, Array2<usize>) {
        let mut weights = Array2::zeros((self.popsize, 2));
        let inv_popsize = 1.0 / (self.popsize - 1) as f64;

        for i in 0..self.popsize {
            let w0 = i as f64 * inv_popsize;
            weights[[i, 0]] = w0;
            weights[[i, 1]] = 1.0 - w0;
        }

        // Optimized neighbor finding with pre-allocation
        let mut neighbors = Array2::zeros((self.popsize, self.niche));
        let mut distances = Vec::with_capacity(self.popsize);

        for i in 0..self.popsize {
            distances.clear();
            let wi = weights.row(i);

            for j in 0..self.popsize {
                let wj = weights.row(j);
                let dist = (wi[0] - wj[0]).powi(2) + (wi[1] - wj[1]).powi(2);
                distances.push((j, dist));
            }

            // Use partial sort for better performance
            distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (k, &(idx, _)) in distances.iter().take(self.niche).enumerate() {
                neighbors[[i, k]] = idx;
            }
        }

        (weights, neighbors)
    }

    /// Optimized initial population generation
    pub fn initial_population(&self, key_node: usize) -> (Vec<Chromosome>, Array1<f64>) {
        let mut chromosomes = Vec::with_capacity(self.popsize);
        let mut ideal_point = Array1::from_elem(2, f64::INFINITY);
        let mut rng = rng();

        // Use cached neighbors
        let neighbors = self.get_neighbors(key_node);

        for _ in 0..self.popsize {
            let community_nodes = if neighbors.is_empty() {
                vec![key_node]
            } else {
                // More intelligent selection strategy
                let num_select = if neighbors.len() <= 2 {
                    neighbors.len()
                } else {
                    rng.random_range(1..=(neighbors.len() / 2).max(1))
                };

                let mut selected = Vec::with_capacity(num_select + 1);
                selected.push(key_node);

                // Use reservoir sampling for better distribution
                for &neighbor in neighbors.choose_multiple(&mut rng, num_select) {
                    selected.push(neighbor);
                }

                selected.sort_unstable();
                selected.dedup();
                selected
            };

            // Calculate objectives
            let obj1 = self.calculate_conductance(&community_nodes);
            let obj2 = self.count_key_nodes_in_community(&community_nodes);

            let chromosome = Chromosome::new(community_nodes, obj1, obj2);
            chromosomes.push(chromosome);

            // Update ideal point
            ideal_point[0] = ideal_point[0].min(obj1);
            ideal_point[1] = ideal_point[1].min(obj2);
        }

        (chromosomes, ideal_point)
    }

    /// Optimized interest calculation with caching
    pub fn calculate_interest(&self, node: usize, community_nodes: &[usize]) -> f64 {
        if community_nodes.contains(&node) {
            return 0.0;
        }

        let community_set: FxHashSet<usize> = community_nodes.iter().cloned().collect();

        // Use cached neighbors and vectorized operations
        let neighbors = self.get_neighbors(node);
        let e_in: f64 = neighbors
            .iter()
            .filter(|&n| community_set.contains(n))
            .map(|&n| self.adj_matrix[[node, n]])
            .sum();

        let e_out: f64 = neighbors
            .iter()
            .filter(|&n| !community_set.contains(n))
            .map(|&n| self.adj_matrix[[node, n]])
            .sum();

        if e_out == 0.0 { 1.0 } else { e_in / e_out }
    }

    pub fn crossover_operator(&self, parent1: &Chromosome, parent2: &Chromosome) -> Vec<usize> {
        let nodes1: FxHashSet<usize> = parent1.nodes.iter().cloned().collect();

        // Pre-allocate with estimated capacity
        let mut child_nodes = Vec::with_capacity(parent1.nodes.len() + 5);
        child_nodes.extend_from_slice(&parent1.nodes);

        let mut rng = rng();

        // Process only different nodes
        for &node in parent2.nodes.iter() {
            if !nodes1.contains(&node) {
                let interest = self.calculate_interest(node, &child_nodes);
                if interest > rng.random::<f64>() {
                    child_nodes.push(node);
                }
            }
        }

        child_nodes
    }

    pub fn mutation_operator(&self, individual_nodes: &mut Vec<usize>, key_node: usize) {
        let mut rng = rng();

        if rng.random::<f64>() > 0.5 {
            let current_set: FxHashSet<usize> = individual_nodes.iter().cloned().collect();
            let mut candidates = Vec::new();

            for &node in individual_nodes.iter() {
                for &neighbor in self.get_neighbors(node) {
                    if !current_set.contains(&neighbor) {
                        candidates.push(neighbor);
                    }
                }
            }

            candidates.sort_unstable();
            candidates.dedup();

            let interest_threshold = 0.5;
            for node in candidates {
                let interest = self.calculate_interest(node, individual_nodes);
                if interest > interest_threshold && interest > rng.random::<f64>() {
                    individual_nodes.push(node);
                }
            }
        } else {
            // Removal operation - preserve key node
            let retention_prob = 0.5;
            individual_nodes
                .retain(|&node| node == key_node || rng.random::<f64>() <= retention_prob);
        }

        // Ensure key node is included
        if !individual_nodes.contains(&key_node) {
            individual_nodes.push(key_node);
        }

        // Efficient deduplication
        individual_nodes.sort_unstable();
        individual_nodes.dedup();
    }

    /// Inlined scalar function for better performance
    #[inline]
    pub fn scalar_func(
        &self,
        obj: &[f64],
        ideal_point: &ArrayView1<f64>,
        weight: &ArrayView1<f64>,
    ) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..2 {
            let diff = (obj[i] - ideal_point[i]).abs();
            let feval = if weight[i] == 0.0 {
                0.00001 * diff
            } else {
                diff * weight[i]
            };
            max_val = max_val.max(feval);
        }
        max_val
    }

    /// Optimized crossover and mutation with better parent selection
    pub fn crossover_mutation(
        &self,
        key_node: usize,
        neighbor_chromosomes: &[Chromosome],
    ) -> Option<Chromosome> {
        if neighbor_chromosomes.is_empty() {
            return None;
        }

        if neighbor_chromosomes.len() == 1 {
            return Some(neighbor_chromosomes[0].clone());
        }

        let mut rng = rng();

        // More efficient parent selection
        let parent1_idx = rng.random_range(0..neighbor_chromosomes.len());
        let mut parent2_idx = rng.random_range(0..neighbor_chromosomes.len());
        while parent2_idx == parent1_idx && neighbor_chromosomes.len() > 1 {
            parent2_idx = rng.random_range(0..neighbor_chromosomes.len());
        }

        let parent1 = &neighbor_chromosomes[parent1_idx];
        let parent2 = &neighbor_chromosomes[parent2_idx];

        // Apply crossover
        let mut child_nodes = if rng.random::<f64>() < self.cross_rate {
            self.crossover_operator(parent1, parent2)
        } else {
            parent1.nodes.clone()
        };

        // Apply mutation
        if rng.random::<f64>() < self.mutate_rate {
            self.mutation_operator(&mut child_nodes, key_node);
        }

        // Calculate objectives
        let obj1 = self.calculate_conductance(&child_nodes);
        let obj2 = self.count_key_nodes_in_community(&child_nodes);

        Some(Chromosome::new(child_nodes, obj1, obj2))
    }

    /// Optimized neighbor update with batch operations
    pub fn update_neighbor(
        &self,
        ideal_point: &mut Array1<f64>,
        chromosomes: &mut [Chromosome],
        child: &Chromosome,
        neighbor_indices: &[usize],
        weights: &ArrayView2<f64>,
    ) {
        let child_obj = [child.obj1, child.obj2];

        // Batch update for better cache locality
        for &i in neighbor_indices {
            let neighbor_obj = [chromosomes[i].obj1, chromosomes[i].obj2];

            let f1 = self.scalar_func(&neighbor_obj, &ideal_point.view(), &weights.row(i));
            let f2 = self.scalar_func(&child_obj, &ideal_point.view(), &weights.row(i));

            if f2 < f1 {
                chromosomes[i] = child.clone();
            }
        }

        // Update ideal point
        ideal_point[0] = ideal_point[0].min(child.obj1);
        ideal_point[1] = ideal_point[1].min(child.obj2);
    }

    /// Run MOEA for a single key node with optimizations
    pub fn run_moea_for_key_node(&self, key_node: usize) -> Vec<Chromosome> {
        let (mut chromosomes, mut ideal_point) = self.initial_population(key_node);
        let (weights, neighbors) = self.init_weights();

        // Pre-allocate vectors to avoid repeated allocations
        let mut neighbor_indices = Vec::with_capacity(self.niche);
        let mut neighbor_chromosomes = Vec::with_capacity(self.niche);

        for _generation in 0..self.max_gen {
            for i in 0..self.popsize {
                neighbor_indices.clear();
                neighbor_chromosomes.clear();

                // Collect neighbor data
                for j in 0..self.niche {
                    let idx = neighbors[[i, j]];
                    neighbor_indices.push(idx);
                    neighbor_chromosomes.push(chromosomes[idx].clone());
                }

                // Generate child
                if let Some(child) = self.crossover_mutation(key_node, &neighbor_chromosomes) {
                    self.update_neighbor(
                        &mut ideal_point,
                        &mut chromosomes,
                        &child,
                        &neighbor_indices,
                        &weights.view(),
                    );
                }
            }
        }

        chromosomes
    }

    pub fn modularity(&self, labels: &[usize]) -> f64 {
        if labels.len() <= 1 {
            return 0.0;
        }

        let m = self.edge_num;
        let mut q = 0.0;

        // Pre-compute community memberships using FxHashMap with capacity hint
        let mut communities: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut community_degrees: FxHashMap<usize, f64> = FxHashMap::default();

        // Single pass to build communities and calculate degrees
        for (i, &label) in labels.iter().enumerate() {
            if label > 0 {
                communities.entry(label).or_insert_with(Vec::new).push(i);
                *community_degrees.entry(label).or_insert(0.0) += self.degree[i];
            }
        }

        // Early exit if only one community
        if communities.len() <= 1 {
            return 0.0;
        }

        // Calculate modularity for each community
        for (label, nodes) in communities {
            if nodes.is_empty() {
                continue;
            }

            let d_c = community_degrees[&label];

            // Optimized internal edges calculation using cached neighbors
            let mut l_c = 0.0;
            for &node in &nodes {
                let neighbors = self.get_neighbors(node);
                for &neighbor in neighbors {
                    // Only count if neighbor is in same community and avoid double counting
                    if labels[neighbor] == label && node <= neighbor {
                        l_c += self.adj_matrix[[node, neighbor]];
                    }
                }
            }

            q += l_c / m - (d_c / (2.0 * m)).powi(2);
        }

        q
    }

    /// Enhanced Community determination EA with improved performance
    pub fn community_determination_ea(
        &self,
        all_communities_by_key_node: &[Vec<Chromosome>],
        ga_generations: usize,
        ga_popsize: usize,
    ) -> (Vec<Chromosome>, f64) {
        if all_communities_by_key_node.is_empty() {
            return (Vec::new(), 0.0);
        }

        let num_key_nodes = all_communities_by_key_node.len();
        let mut rng = rng();

        // Initialize population with better diversity and pre-allocation
        let mut population: Vec<Vec<usize>> = Vec::with_capacity(ga_popsize);
        for _ in 0..ga_popsize {
            let individual: Vec<usize> = (0..num_key_nodes)
                .map(|i| {
                    let len = all_communities_by_key_node[i].len();
                    if len > 0 { rng.random_range(0..len) } else { 0 }
                })
                .collect();
            population.push(individual);
        }

        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_individual: Option<Vec<usize>> = None;

        // Pre-allocate fitness vector
        let mut fitness_values = Vec::with_capacity(ga_popsize);

        for _generation in 0..ga_generations {
            // Parallel fitness evaluation
            fitness_values.clear();
            fitness_values.par_extend(population.par_iter().map(|individual| {
                self.evaluate_community_combination(individual, all_communities_by_key_node)
            }));

            // Update best solution
            for (i, &fitness) in fitness_values.iter().enumerate() {
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_individual = Some(population[i].clone());
                }
            }

            // Enhanced genetic operators
            population = self.enhanced_ga_operators(
                population,
                &fitness_values,
                all_communities_by_key_node,
            );
        }

        // Extract best communities
        let best_communities = if let Some(ref best_ind) = best_individual {
            best_ind
                .iter()
                .enumerate()
                .filter_map(|(i, &selection)| {
                    if i < all_communities_by_key_node.len()
                        && selection < all_communities_by_key_node[i].len()
                    {
                        Some(all_communities_by_key_node[i][selection].clone())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        (best_communities, best_fitness)
    }

    /// Optimized fitness evaluation with better memory management
    pub fn evaluate_community_combination(
        &self,
        individual: &[usize],
        all_communities_by_key_node: &[Vec<Chromosome>],
    ) -> f64 {
        // Pre-allocate labels vector
        let mut labels = vec![0; self.num_nodes];
        let mut community_id = 1;

        // More efficient community assignment
        for (i, &selection) in individual.iter().enumerate() {
            if i < all_communities_by_key_node.len()
                && selection < all_communities_by_key_node[i].len()
            {
                for &node in &all_communities_by_key_node[i][selection].nodes {
                    if node < self.num_nodes {
                        labels[node] = community_id;
                    }
                }
                community_id += 1;
            }
        }

        self.modularity(&labels)
    }

    /// Enhanced genetic algorithm operators with better performance
    pub fn enhanced_ga_operators(
        &self,
        population: Vec<Vec<usize>>,
        fitness_values: &[f64],
        all_communities_by_key_node: &[Vec<Chromosome>],
    ) -> Vec<Vec<usize>> {
        let mut new_population = Vec::with_capacity(population.len());
        let mut rng = rng();

        // Elitism - keep top performers
        if !fitness_values.is_empty() {
            let elite_count = (population.len() / 10).max(1).min(5);
            let mut indexed_fitness: Vec<(usize, f64)> = fitness_values
                .iter()
                .enumerate()
                .map(|(i, &f)| (i, f))
                .collect();
            indexed_fitness.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (idx, _) in indexed_fitness.iter().take(elite_count) {
                new_population.push(population[*idx].clone());
            }
        }

        // Improved selection with fitness scaling
        let min_fitness = fitness_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_fitness = fitness_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let fitness_range = max_fitness - min_fitness;

        let adjusted_fitness: Vec<f64> = if fitness_range > 0.0 {
            fitness_values
                .iter()
                .map(|&f| ((f - min_fitness) / fitness_range) + 0.1)
                .collect()
        } else {
            vec![1.0; fitness_values.len()]
        };

        let fitness_sum: f64 = adjusted_fitness.iter().sum();
        let probabilities: Vec<f64> = adjusted_fitness.iter().map(|&f| f / fitness_sum).collect();

        // Generate offspring with improved operators
        while new_population.len() < population.len() {
            let child = if rng.random::<f64>() < 0.8 {
                // Tournament selection for parents
                let tournament_size = 3;
                let parent1_idx =
                    self.tournament_selection(&probabilities, tournament_size, &mut rng);
                let parent2_idx =
                    self.tournament_selection(&probabilities, tournament_size, &mut rng);

                let parent1 = &population[parent1_idx];
                let parent2 = &population[parent2_idx];

                // Multi-point crossover
                let mut child = parent1.clone();
                let crossover_points = 2;
                for _ in 0..crossover_points {
                    let point = rng.random_range(0..parent1.len());
                    child[point] = parent2[point];
                }
                child
            } else {
                // Direct selection
                let dist = WeightedIndex::new(&probabilities).unwrap();
                let parent_idx = dist.sample(&mut rng);
                population[parent_idx].clone()
            };

            // Adaptive mutation
            let mut mutated_child = child;
            let mutation_rate = if fitness_range > 0.1 {
                self.mutate_rate
            } else {
                self.mutate_rate * 2.0
            };

            for i in 0..mutated_child.len() {
                if rng.random::<f64>() < mutation_rate && i < all_communities_by_key_node.len() {
                    let max_selection = all_communities_by_key_node[i].len();
                    if max_selection > 0 {
                        mutated_child[i] = rng.random_range(0..max_selection);
                    }
                }
            }

            new_population.push(mutated_child);
        }

        new_population.truncate(population.len());
        new_population
    }

    /// Tournament selection for better diversity
    fn tournament_selection(
        &self,
        probabilities: &[f64],
        tournament_size: usize,
        rng: &mut ThreadRng,
    ) -> usize {
        let mut best_idx = rng.random_range(0..probabilities.len());
        let mut best_fitness = probabilities[best_idx];

        for _ in 1..tournament_size {
            let idx = rng.random_range(0..probabilities.len());
            if probabilities[idx] > best_fitness {
                best_idx = idx;
                best_fitness = probabilities[idx];
            }
        }

        best_idx
    }

    /// Optimized community merging
    pub fn merge_overlapping_communities(&self, communities: &[Chromosome]) -> Vec<Community> {
        if communities.is_empty() {
            return Vec::new();
        }

        let mut merged_communities = Vec::new();
        let mut used = vec![false; communities.len()];

        for (i, comm1) in communities.iter().enumerate() {
            if used[i] {
                continue;
            }

            let mut current_community: FxHashSet<usize> = comm1.nodes.iter().cloned().collect();
            used[i] = true;

            // Check for overlaps with remaining communities
            for (j, comm2) in communities.iter().enumerate().skip(i + 1) {
                if used[j] {
                    continue;
                }

                let nodes2: FxHashSet<usize> = comm2.nodes.iter().cloned().collect();
                let intersection: FxHashSet<usize> =
                    current_community.intersection(&nodes2).cloned().collect();

                if intersection.is_empty() {
                    continue;
                }

                // Enhanced similarity calculation
                let union_size = current_community.union(&nodes2).count();
                let similarity =
                    intersection.len() as f64 / current_community.len().min(nodes2.len()) as f64;
                let jaccard = intersection.len() as f64 / union_size as f64;

                // Use both similarity measures
                if similarity > self.alpha || jaccard > self.alpha * 0.5 {
                    // Merge communities
                    current_community.extend(nodes2);
                    used[j] = true;
                } else {
                    // Enhanced conflict resolution - assign to better fitting community
                    for &node in &intersection {
                        let mut temp_comm1 = current_community.clone();
                        temp_comm1.remove(&node);
                        let mut temp_comm2 = nodes2.clone();
                        temp_comm2.remove(&node);

                        let fit1 = self.calculate_community_fit(
                            node,
                            &temp_comm1.into_iter().collect::<Vec<_>>(),
                        );
                        let fit2 = self.calculate_community_fit(
                            node,
                            &temp_comm2.into_iter().collect::<Vec<_>>(),
                        );

                        if fit1 < fit2 {
                            current_community.remove(&node);
                        }
                    }
                }
            }

            if !current_community.is_empty() {
                merged_communities.push(Community {
                    nodes: current_community.into_iter().collect(),
                });
            }
        }

        merged_communities
    }

    /// Optimized community fit calculation
    #[inline]
    pub fn calculate_community_fit(&self, node: usize, community_nodes: &[usize]) -> f64 {
        if community_nodes.is_empty() {
            return 0.0;
        }

        let internal_edges: f64 = community_nodes
            .iter()
            .map(|&i| self.adj_matrix[[node, i]])
            .sum();

        let total_edges = self.degree[node];
        let external_edges = total_edges - internal_edges;

        if external_edges == 0.0 {
            1.0
        } else {
            internal_edges / total_edges
        }
    }

    /// Optimized assignment of remaining nodes with batch processing
    pub fn assign_remaining_nodes(&self, labels: &mut [usize]) {
        let unassigned: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 0 { Some(i) } else { None })
            .collect();

        // Pre-compute community sizes for load balancing using FxHashMap
        let mut community_sizes: FxHashMap<usize, usize> = FxHashMap::default();
        for &label in labels.iter() {
            if label > 0 {
                *community_sizes.entry(label).or_insert(0) += 1;
            }
        }

        for node in unassigned {
            // Use cached neighbors
            let neighbors = self.get_neighbors(node);

            let neighbor_labels: Vec<usize> = neighbors
                .iter()
                .map(|&i| labels[i])
                .filter(|&label| label > 0)
                .collect();

            if !neighbor_labels.is_empty() {
                // Weighted assignment based on edge weights and community sizes using FxHashMap
                let mut label_weights: FxHashMap<usize, f64> = FxHashMap::default();

                for &neighbor in neighbors {
                    let neighbor_label = labels[neighbor];
                    if neighbor_label > 0 {
                        let edge_weight = self.adj_matrix[[node, neighbor]];
                        let community_size = community_sizes.get(&neighbor_label).unwrap_or(&1);
                        // Prefer smaller communities for better balance
                        let size_penalty = 1.0 / (*community_size as f64).sqrt();
                        *label_weights.entry(neighbor_label).or_insert(0.0) +=
                            edge_weight * size_penalty;
                    }
                }

                let best_community = *label_weights
                    .iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;

                labels[node] = best_community;
                *community_sizes.entry(best_community).or_insert(0) += 1;
            } else {
                // Assign to smallest existing community or create new one
                if let Some((&smallest_comm, _)) =
                    community_sizes.iter().min_by_key(|(_, size)| **size)
                {
                    labels[node] = smallest_comm;
                    *community_sizes.entry(smallest_comm).or_insert(0) += 1;
                } else {
                    labels[node] = 1;
                    community_sizes.insert(1, 1);
                }
            }
        }
    }

    pub fn run(&mut self) -> FxHashMap<usize, usize> {
        let key_nodes = self.find_key_nodes();
        println!("Found {} key nodes", key_nodes.len());

        if key_nodes.is_empty() {
            let mut partition = FxHashMap::default();
            for (i, &original_node) in self.active_nodes.iter().enumerate() {
                partition.insert(original_node, i);
            }
            return partition;
        }

        println!("Step 2: Running MOEA for each key node in parallel...");
        let start_time = std::time::Instant::now();

        let chunk_size = (key_nodes.len() / rayon::current_num_threads()).max(1);
        let all_communities_by_key_node: Vec<Vec<Chromosome>> = key_nodes
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .par_iter()
                    .map(|&key_node| self.run_moea_for_key_node(key_node))
            })
            .collect();

        println!(
            "Parallel processing completed in {:.2} seconds",
            start_time.elapsed().as_secs_f64()
        );

        // Step 3: Community determination using EA with adaptive parameters
        println!("Step 3: Determining final communities using EA...");
        let (selected_communities, best_modularity) = self.community_determination_ea(
            &all_communities_by_key_node,
            std::cmp::min(50, key_nodes.len() * 2), // Adaptive generations
            std::cmp::min(50, key_nodes.len() * 3), // Adaptive population size
        );

        // Step 4: Enhanced post-processing
        println!("Step 4: Post-processing communities...");
        let merged_communities = self.merge_overlapping_communities(&selected_communities);

        // Create final labels with pre-allocation
        let mut labels = vec![0; self.num_nodes];
        let mut community_id = 1;

        for community in &merged_communities {
            for &node in &community.nodes {
                if node < self.num_nodes {
                    labels[node] = community_id;
                }
            }
            community_id += 1;
        }

        // Assign remaining nodes with enhanced method
        self.assign_remaining_nodes(&mut labels);

        // Convert to final result format with original node mapping using FxHashMap
        let mut partition =
            FxHashMap::with_capacity_and_hasher(self.active_nodes.len(), Default::default());
        for (i, &original_node) in self.active_nodes.iter().enumerate() {
            partition.insert(original_node, labels[i]);
        }

        // Add isolated nodes as separate communities if any exist
        let all_original_nodes: FxHashSet<usize> = (0..self.active_nodes.len() + 100).collect(); // Adjust based on your graph size
        let active_set: FxHashSet<usize> = self.active_nodes.iter().cloned().collect();
        let isolated_nodes: Vec<usize> = all_original_nodes
            .difference(&active_set)
            .cloned()
            .collect();

        let mut max_community_id = partition.values().max().cloned().unwrap_or(0);
        for isolated_node in isolated_nodes {
            max_community_id += 1;
            partition.insert(isolated_node, max_community_id);
        }

        let unique_communities = partition.values().collect::<FxHashSet<_>>().len();
        let total_time = start_time.elapsed().as_secs_f64();

        println!("Final result: {} communities detected", unique_communities);
        println!("Total execution time: {:.2} seconds", total_time);
        println!("Best modularity: {:.4}", best_modularity);

        partition
    }
}
