use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use crate::{
    graph::{Graph, NodeId}, 
    utils::*
};
/// Represents a network as an adjacency list
struct Network {
    nodes: usize,
    edges: Vec<(usize, usize)>,
    adjacency_list: Vec<Vec<usize>>,
    degree: Vec<usize>,
}

impl Network {
    /// Create a new network from a list of edges
    fn new(nodes: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut adjacency_list = vec![Vec::new(); nodes];
        let mut degree = vec![0; nodes];
        
        for &(u, v) in &edges {
            adjacency_list[u].push(v);
            adjacency_list[v].push(u);
            degree[u] += 1;
            degree[v] += 1;
        }
        
        Network {
            nodes,
            edges,
            adjacency_list,
            degree,
        }
    }

    fn from_graph(graph: Graph) -> Self {
           // Create a sorted vector of nodes to ensure deterministic ordering.
    let mut nodes_vec: Vec<NodeId> = graph.nodes.iter().cloned().collect();
    nodes_vec.sort();

    // Build a mapping from original NodeId to new index (usize).
    let mapping: HashMap<NodeId, usize> = nodes_vec
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    // Convert edges using the mapping.
    let edges = graph
        .edges
        .iter()
        .map(|(u, v)| {
            let new_u = *mapping.get(u).expect("Node should exist in mapping");
            let new_v = *mapping.get(v).expect("Node should exist in mapping");
            (new_u, new_v)
        })
        .collect();

    // Build the new adjacency list vector.
    let mut new_adj_list = vec![Vec::new(); nodes_vec.len()];
    for (&node, neighbors) in graph.adjacency_list.iter() {
        let i = mapping[&node];
        for neighbor in neighbors {
            // Convert each neighbor using the mapping.
            new_adj_list[i].push(mapping[neighbor]);
        }
    }

    // Compute the degree for each node.
    let degree = new_adj_list.iter().map(|neighbors| neighbors.len()).collect();

    Network {
        nodes: nodes_vec.len(),
        edges,
        adjacency_list: new_adj_list,
        degree,
    } 
    }
    
    /// Get total number of edges in the network
    fn total_edges(&self) -> usize {
        self.edges.len()
    }
    
    /// Extract a subnetwork based on a set of nodes
    fn extract_subnetwork(&self, nodes: &HashSet<usize>) -> Network {
        let mut edges = Vec::new();
        let mut node_mapping: HashMap<usize, usize> = HashMap::new();
        
        // Create a mapping from original node IDs to sequential IDs for the subnetwork
        for (new_id, &orig_id) in nodes.iter().enumerate() {
            node_mapping.insert(orig_id, new_id);
        }
        
        // Keep edges where both endpoints are in the subnetwork
        for &(u, v) in &self.edges {
            if nodes.contains(&u) && nodes.contains(&v) {
                edges.push((node_mapping[&u], node_mapping[&v]));
            }
        }
        
        Network::new(nodes.len(), edges)
    }
}

/// Represents an individual in the population for community detection
#[derive(Clone)]
struct Individual {
    genes: Vec<usize>,  // Community identifiers for each node
    fitness: f64,       // Modularity value
}

impl Individual {
    /// Create a new random individual
    fn new_random(network: &Network, rng: &mut ThreadRng) -> Self {
        let n = network.nodes;
        let mut genes = vec![0; n];
        
        // Assign each node to a random community
        for i in 0..n {
            genes[i] = rng.random_range(0..n);
        }
        
        // Initialize some nodes to have same community as their neighbors
        let num_nodes_to_process = (n as f64 * 0.1).ceil() as usize;
        let selected_nodes: Vec<usize> = (0..n).collect::<Vec<_>>()
            .choose_multiple(rng, num_nodes_to_process).cloned().collect();
            
        for &node in &selected_nodes {
            if !network.adjacency_list[node].is_empty() {
                let neighbor = *network.adjacency_list[node].choose(rng).unwrap();
                genes[node] = genes[neighbor];
            }
        }
        
        let mut individual = Individual {
            genes,
            fitness: 0.0,
        };
        individual.fitness = compute_modularity(&individual, network);
        individual
    }
    
    /// Extract genes for a specific subnetwork
    fn extract_subgenes(&self, nodes: &HashSet<usize>) -> Vec<usize> {
        let mut subgenes = Vec::new();
        let mut seen_communities = HashMap::new();
        let mut next_comm_id = 0;
        
        for &node in nodes {
            let comm_id = self.genes[node];
            let new_comm_id = *seen_communities.entry(comm_id).or_insert_with(|| {
                let id = next_comm_id;
                next_comm_id += 1;
                id
            });
            subgenes.push(new_comm_id);
        }
        
        subgenes
    }
    
    /// Update genes from a subnetwork solution
    fn update_from_subnetwork(&mut self, subnetwork_nodes: &Vec<usize>, best_subgenes: &Vec<usize>) {
        let mut comm_mapping = HashMap::new();
        let mut next_comm_id = 0;
        
        // Find the highest community ID currently in use
        for &comm_id in &self.genes {
            next_comm_id = next_comm_id.max(comm_id + 1);
        }
        
        // Map subnetwork community IDs to new unique IDs in the full network
        for i in 0..subnetwork_nodes.len() {
            let original_node = subnetwork_nodes[i];
            let subnetwork_comm = best_subgenes[i];
            
            if !comm_mapping.contains_key(&subnetwork_comm) {
                comm_mapping.insert(subnetwork_comm, next_comm_id);
                next_comm_id += 1;
            }
            
            self.genes[original_node] = comm_mapping[&subnetwork_comm];
        }
    }
}

/// Compute the modularity of a network partition
fn compute_modularity(individual: &Individual, network: &Network) -> f64 {
    let n = network.nodes;
    let m = network.total_edges() as f64;
    
    if m == 0.0 {
        return 0.0;
    }
    
    // Create a mapping of communities and nodes they contain
    let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        communities.entry(individual.genes[i])
            .or_insert_with(Vec::new)
            .push(i);
    }
    
    let mut q = 0.0;
    
    for (_comm_id, nodes) in communities {
        let mut l_c = 0; // Number of edges within community
        let mut d_c = 0; // Sum of degrees of nodes in community
        
        for &i in &nodes {
            d_c += network.degree[i];
            
            for &j in &network.adjacency_list[i] {
                if nodes.contains(&j) && i < j { // Count each edge only once
                    l_c += 1;
                }
            }
        }
        
        q += (l_c as f64 / m) - ((d_c as f64 / (2.0 * m)).powi(2));
    }
    
    q
}

/// Differential Evolution operator (DE/rand/1/bin)
fn differential_evolution(
    population: &mut Vec<Individual>,
    network: &Network,
    f: f64,
    cr: f64,
    rng: &mut ThreadRng
) {
    let pop_size = population.len();
    let dimension = network.nodes;
    let mut new_population = Vec::with_capacity(pop_size);
    
    for i in 0..pop_size {
        let target = &population[i];
        
        // Select three distinct random individuals different from target
        let mut indices: Vec<usize> = (0..pop_size).filter(|&x| x != i).collect();
        indices.shuffle(rng);
        let r1 = indices[0];
        let r2 = indices[1];
        let r3 = indices[2];
        
        // Create trial vector using DE/rand/1/bin strategy
        let mut trial = target.clone();
        
        let j_rand = rng.random_range(0..dimension);
        for j in 0..dimension {
            if rng.random::<f64>() < cr || j == j_rand {
                // Create new community ID based on DE formula
                let mut new_comm_id = population[r1].genes[j];
                
                if rng.random::<f64>() < f {  // Apply scaling factor F probabilistically
                    // Different communities
                    if population[r2].genes[j] != population[r3].genes[j] {
                        // Choose one of them randomly
                        new_comm_id = if rng.random::<bool>() {
                            population[r2].genes[j]
                        } else {
                            population[r3].genes[j]
                        };
                    }
                }
                
                trial.genes[j] = new_comm_id;
            }
        }
        
        // Evaluate trial vector
        trial.fitness = compute_modularity(&trial, network);
        
        // Selection
        if trial.fitness >= target.fitness {
            new_population.push(trial);
        } else {
            new_population.push(target.clone());
        }
    }
    
    *population = new_population;
}

/// Local Moving scheme to optimize community structure
fn local_moving_scheme(individual: &mut Individual, network: &Network) -> bool {
    let n = network.nodes;
    let mut improved = false;
    let mut continue_moving = true;
    
    while continue_moving {
        continue_moving = false;
        
        // Process nodes in random order
        let mut node_indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::rng();
        node_indices.shuffle(&mut rng);
        
        for &node in &node_indices {
            let original_comm = individual.genes[node];
            let mut best_comm = original_comm;
            let mut best_delta_q = 0.0;
            
            // Try moving node to each of its neighbors' communities
            let mut neighbor_comms = HashSet::new();
            for &neighbor in &network.adjacency_list[node] {
                neighbor_comms.insert(individual.genes[neighbor]);
            }
            
            for &comm in &neighbor_comms {
                if comm == original_comm {
                    continue;
                }
                
                // Move node to this community
                individual.genes[node] = comm;
                let new_q = compute_modularity(individual, network);
                let delta_q = new_q - individual.fitness;
                
                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_comm = comm;
                }
            }
            
            // Restore original community for further comparisons
            individual.genes[node] = original_comm;
            
            // Apply best move if it improves modularity
            if best_delta_q > 0.0 {
                individual.genes[node] = best_comm;
                individual.fitness += best_delta_q;
                improved = true;
                continue_moving = true;
            }
        }
    }
    
    improved
}

/// Adapted Kernighan-Lin moving scheme to optimize network partition
fn adapted_kl_moving_scheme(individual: &mut Individual, network: &Network) -> bool {
    let n = network.nodes;
    let mut improved = false;
    let mut continue_outer = true;
    
    while continue_outer {
        continue_outer = false;
        let mut best_partition = individual.clone();
        let mut best_q = individual.fitness;
        let k = (10.0 * (n as f64).log2()) as usize;
        let mut moves_without_improvement = 0;
        
        let mut node_indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::rng();
        node_indices.shuffle(&mut rng);
        
        for &node in &node_indices {
            let original_comm = individual.genes[node];
            
            // Find the best move for this node (to any community)
            let mut best_move_comm = original_comm;
            let mut best_move_q = -std::f64::INFINITY;
            
            // Consider all neighbor communities
            let mut neighbor_comms = HashSet::new();
            neighbor_comms.insert(original_comm);  // Include current community
            
            for &neighbor in &network.adjacency_list[node] {
                neighbor_comms.insert(individual.genes[neighbor]);
            }
            
            for &comm in &neighbor_comms {
                // Apply move
                individual.genes[node] = comm;
                let q = compute_modularity(individual, network);
                
                if q > best_move_q {
                    best_move_q = q;
                    best_move_comm = comm;
                }
            }
            
            // Apply best move
            individual.genes[node] = best_move_comm;
            individual.fitness = best_move_q;
            
            // Update global best if improved
            if individual.fitness > best_q {
                best_partition = individual.clone();
                best_q = individual.fitness;
                moves_without_improvement = 0;
                improved = true;
                continue_outer = true;
            } else {
                moves_without_improvement += 1;
                if moves_without_improvement >= k {
                    break;
                }
            }
        }
        
        // Continue from best found partition
        *individual = best_partition;
    }
    
    improved
}

/// Node grouping scheme to split network into subnetworks
fn node_grouping_scheme(
    network: &Network,
    num_subnetworks: usize,
    rng: &mut ThreadRng
) -> Vec<Vec<usize>> {
    let n = network.nodes;
    let nodes_per_subnetwork = n / num_subnetworks;
    
    // Initial random grouping
    let mut all_nodes: Vec<usize> = (0..n).collect();
    all_nodes.shuffle(rng);
    
    let mut subnetworks: Vec<Vec<usize>> = Vec::with_capacity(num_subnetworks);
    for i in 0..num_subnetworks {
        let start = i * nodes_per_subnetwork;
        let end = if i == num_subnetworks - 1 { n } else { (i + 1) * nodes_per_subnetwork };
        subnetworks.push(all_nodes[start..end].to_vec());
    }
    
    // Optimize subnetworks using local moving
    let mut optimized_subnetworks: Vec<Vec<usize>> = Vec::with_capacity(num_subnetworks);
    
    for subnetwork_nodes in &subnetworks {
        let subnetwork_set: HashSet<usize> = subnetwork_nodes.iter().cloned().collect();
        let subnetwork = network.extract_subnetwork(&subnetwork_set);
        
        // Create initial random solution for this subnetwork
        let mut individual = Individual::new_random(&subnetwork, rng);
        
        // Optimize using local moving
        local_moving_scheme(&mut individual, &subnetwork);
        
        // Group nodes by communities
        let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..subnetwork_nodes.len() {
            communities.entry(individual.genes[i])
                .or_insert_with(Vec::new)
                .push(subnetwork_nodes[i]);
        }
        
        // Add each community as a group
        for (_comm_id, nodes) in communities {
            if !nodes.is_empty() {
                optimized_subnetworks.push(nodes);
            }
        }
    }
    
    // Ensure we have exactly num_subnetworks groups
    while optimized_subnetworks.len() > num_subnetworks {
        // Merge the two smallest groups
        optimized_subnetworks.sort_by(|a, b| a.len().cmp(&b.len()));
        let g1 = optimized_subnetworks.remove(0);
        let g2 = optimized_subnetworks.remove(0);
        let mut merged = g1;
        merged.extend(g2);
        optimized_subnetworks.push(merged);
    }
    
    while optimized_subnetworks.len() < num_subnetworks {
        // Split the largest group
        optimized_subnetworks.sort_by(|a, b| b.len().cmp(&a.len()));
        let largest = optimized_subnetworks.remove(0);
        let split_point = largest.len() / 2;
        
        let g1 = largest[0..split_point].to_vec();
        let g2 = largest[split_point..].to_vec();
        
        optimized_subnetworks.push(g1);
        optimized_subnetworks.push(g2);
    }
    
    optimized_subnetworks
}

/// Main CoCoMi algorithm
fn cocomi(
    network: &Network,
    max_generations: usize,
    pop_size: usize,
    num_subnetworks: usize,
    subnetwork_generations: usize,
    whole_network_generations: usize,
    f: f64,
    cr: f64,
) -> Individual {
    let mut generation = 0;
    let mut rng = rand::rng();
    
    // Initialize population
    let mut population: Vec<Individual> = (0..pop_size)
        .map(|_| Individual::new_random(network, &mut rng))
        .collect();
    
    // Sort population by fitness
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
    
    while generation < max_generations {
        // Split network into subnetworks
        let subnetworks = node_grouping_scheme(network, num_subnetworks, &mut rng);
        
        // Optimize each subnetwork
        for subnetwork_nodes in &subnetworks {
            let subnetwork_set: HashSet<usize> = subnetwork_nodes.iter().cloned().collect();
            let subnetwork = network.extract_subnetwork(&subnetwork_set);
            
            // Extract corresponding genes for each individual
            let mut subpopulation: Vec<Individual> = Vec::with_capacity(pop_size);
            
            for individual in &population {
                let subgenes = individual.extract_subgenes(&subnetwork_set);
                let mut subindividual = Individual {
                    genes: subgenes,
                    fitness: 0.0,
                };
                subindividual.fitness = compute_modularity(&subindividual, &subnetwork);
                subpopulation.push(subindividual);
            }
            
            // Optimize subpopulation with DE and local moving
            for _ in 0..subnetwork_generations {
                differential_evolution(&mut subpopulation, &subnetwork, f, cr, &mut rng);
                
                // Apply local moving to best individual
                subpopulation.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
                local_moving_scheme(&mut subpopulation[0], &subnetwork);
            }
            
            // Update main population with best subnetwork solution
            let best_subindividual = &subpopulation[0];
            
            for individual in &mut population {
                individual.update_from_subnetwork(subnetwork_nodes, &best_subindividual.genes);
                individual.fitness = compute_modularity(individual, network);
            }
            
            // Update generation count
            generation += subnetwork_generations;
            if generation >= max_generations {
                break;
            }
        }
        
        // Sort population by fitness
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
        
        // Optimize whole network using adapted KL moving scheme
        for _ in 0..whole_network_generations {
            adapted_kl_moving_scheme(&mut population[0], network);
        }
        
        // Update generation count
        generation += whole_network_generations;
    }
    
    // Return best individual
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
    population[0].clone()
}

/// Recursive partitioning scheme to address resolution limit problem
fn recursive_partitioning(
    network: &Network,
    q_min: f64,
    z_score_threshold: f64,
    cocomi_params: (usize, usize, usize, usize, usize, f64, f64),
) -> Vec<Vec<usize>> {
    let (max_generations, pop_size, num_subnetworks, 
         subnetwork_generations, whole_network_generations, f, cr) = cocomi_params;
    
    // Run CoCoMi on the network
    let best_individual = cocomi(
        network,
        max_generations,
        pop_size,
        num_subnetworks,
        subnetwork_generations,
        whole_network_generations,
        f,
        cr,
    );
    
    // Extract communities
    let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..network.nodes {
        communities.entry(best_individual.genes[i])
            .or_insert_with(Vec::new)
            .push(i);
    }
    
    let mut result: Vec<Vec<usize>> = Vec::new();
    
    // Process each community
    for (_comm_id, nodes) in communities {
        if nodes.len() <= 2 {
            // Communities with 1 or 2 nodes can't be further partitioned
            result.push(nodes);
            continue;
        }
        
        let subnetwork_set: HashSet<usize> = nodes.iter().cloned().collect();
        let subnetwork = network.extract_subnetwork(&subnetwork_set);
        
        // Run CoCoMi on the subnetwork to check modularity
        let subnetwork_solution = cocomi(
            &subnetwork,
            max_generations / 2,  // Use fewer generations for subnetworks
            pop_size,
            std::cmp::max(2, num_subnetworks / 2),
            subnetwork_generations,
            whole_network_generations,
            f,
            cr,
        );
        
        let subnetwork_q = subnetwork_solution.fitness;
        
        if subnetwork_q < q_min {
            // No strong community structure, don't partition further
            result.push(nodes);
            continue;
        }
        
        // Check statistical significance using Monte Carlo method
        let mut random_q_values = Vec::new();
        let mut rng = rand::rng();
        const NUM_RANDOM_NETWORKS: usize = 10;  // Use a smaller value for performance
        
        for _ in 0..NUM_RANDOM_NETWORKS {
            // Generate random network with same degree distribution
            let mut random_edges = Vec::new();
            let mut node_stubs: Vec<usize> = Vec::new();
            
            for i in 0..subnetwork.nodes {
                for _ in 0..subnetwork.degree[i] {
                    node_stubs.push(i);
                }
            }
            
            node_stubs.shuffle(&mut rng);
            
            for i in (0..node_stubs.len()).step_by(2) {
                if i + 1 < node_stubs.len() {
                    let u = node_stubs[i];
                    let v = node_stubs[i + 1];
                    if u != v {  // Avoid self-loops
                        random_edges.push((u, v));
                    }
                }
            }
            
            let random_network = Network::new(subnetwork.nodes, random_edges);
            
            // Run CoCoMi on random network
            let random_solution = cocomi(
                &random_network,
                max_generations / 4,  // Use even fewer generations for random networks
                pop_size,
                std::cmp::max(2, num_subnetworks / 2),
                subnetwork_generations / 2,
                whole_network_generations / 2,
                f,
                cr,
            );
            
            random_q_values.push(random_solution.fitness);
        }
        
        // Calculate Z-score
        let mean_q: f64 = random_q_values.iter().sum::<f64>() / (NUM_RANDOM_NETWORKS as f64);
        let variance_q: f64 = random_q_values.iter()
            .map(|&q| (q - mean_q).powi(2))
            .sum::<f64>() / (NUM_RANDOM_NETWORKS as f64);
        let std_dev_q = variance_q.sqrt();
        
        let z_score = if std_dev_q > 0.0 {
            (subnetwork_q - mean_q) / std_dev_q
        } else {
            0.0
        };
        
        if z_score >= z_score_threshold {
            // Significant community structure, recursively partition
            let sub_communities = recursive_partitioning(
                &subnetwork,
                q_min,
                z_score_threshold,
                cocomi_params,
            );
            
            // Map subnetwork node IDs back to original network IDs
            for sub_comm in sub_communities {
                let original_ids: Vec<usize> = sub_comm.iter()
                    .map(|&idx| nodes[idx])
                    .collect();
                result.push(original_ids);
            }
        } else {
            // Not significant, keep as is
            result.push(nodes);
        }
    }
    
    result
}


#[pyclass]
pub struct CoCoMi {
    network: Network,
}

#[pymethods]
impl CoCoMi {
    #[new]
    #[pyo3(signature = (graph))]
    pub fn new(graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        /* Convert from networkx to graph */
        let edges = get_edges(graph)?;
        let graph = build_graph(edges);
        // Convert the graph to a network
        let network = Network::from_graph(graph);
        Ok(CoCoMi { network })
    }

    #[pyo3(signature = ())]
    pub fn run(&self) -> PyResult<Vec<Vec<usize>>> {
        // Use the network data provided in self.network.
        let n = self.network.nodes;
        
        // Optionally print some information from the network:
        //println!("Using network with {} nodes and {} edges", n, self.network.edges.len());
        
        // Set CoCoMi parameters
        let max_generations = 100;
        let pop_size = 50;
        let num_subnetworks = 1;
        let subnetwork_generations = 0;
        let whole_network_generations = 10;
        let f = 0.9;  // DE scaling factor
        let cr = 0.9; // DE crossover rate
        
        // Run CoCoMi using the network from self.network
        let best_individual = cocomi(
            &self.network,
            max_generations,
            pop_size,
            num_subnetworks,
            subnetwork_generations,
            whole_network_generations,
            f,
            cr,
        );
        
        // Print the best modularity value
        println!("Best modularity: {}", best_individual.fitness);
        
        // Extract communities from best_individual.genes
        let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            communities
                .entry(best_individual.genes[i])
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        // Run recursive partitioning for higher resolution.
        let cocomi_params = (
            max_generations,
            pop_size,
            num_subnetworks,
            subnetwork_generations,
            whole_network_generations,
            f,
            cr,
        );
        
        let final_communities = recursive_partitioning(
            &self.network,
            0.3,  // q_min
            2.0,  // z_score threshold
            cocomi_params,
        );
        
        // Return the final communities as a Vec<Vec<usize>>
        Ok(final_communities)
    }
}
