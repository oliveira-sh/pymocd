//! Partition utilities and community normalization
//! 
//! This module provides utilities for handling network partitions (community assignments)
//! and normalizing community IDs for consistent output.

use rustc_hash::FxHashMap;
use super::graph::{Graph, NodeId, CommunityId};

/// Represents a partition of a graph into communities
pub type Partition = FxHashMap<NodeId, CommunityId>;

/// Normalizes community IDs to ensure they start from 0 and are consecutive
/// 
/// Isolated nodes (degree 0) are assigned community ID -1.
/// 
/// # Arguments
/// * `graph` - The graph being partitioned
/// * `partition` - The partition to normalize
/// 
/// # Returns
/// A new partition with normalized community IDs
/// 
/// # Examples
/// 
/// ```rust
/// use rustc_hash::FxHashMap;
/// use pymocd::graph::{Graph, normalize_community_ids};
/// 
/// let graph = Graph::new();
/// let mut partition = FxHashMap::default();
/// partition.insert(0, 5);
/// partition.insert(1, 5);
/// partition.insert(2, 10);
/// 
/// let normalized = normalize_community_ids(&graph, partition);
/// // Community 5 becomes 0, community 10 becomes 1
/// ```
pub fn normalize_community_ids(graph: &Graph, partition: Partition) -> Partition {
    let mut community_mapping: FxHashMap<CommunityId, CommunityId> = FxHashMap::default();
    let mut next_community_id: CommunityId = 0;
    let mut normalized_partition: Partition = FxHashMap::default();

    // First pass: create mapping for existing communities
    for &original_community in partition.values() {
        if !community_mapping.contains_key(&original_community) {
            community_mapping.insert(original_community, next_community_id);
            next_community_id += 1;
        }
    }

    // Second pass: apply normalization and handle isolated nodes
    for node in graph.nodes_iter() {
        let normalized_community = if graph.degree(node) == 0 {
            -1 // Isolated nodes get community -1
        } else {
            match partition.get(node) {
                Some(&original_community) => community_mapping[&original_community],
                None => {
                    // Node not in partition, assign new community
                    let new_community = next_community_id;
                    next_community_id += 1;
                    new_community
                }
            }
        };
        
        normalized_partition.insert(*node, normalized_community);
    }

    normalized_partition
}

/// Calculates the modularity of a partition
/// 
/// Modularity measures the strength of division of a network into communities.
/// Values closer to 1 indicate stronger community structure.
/// 
/// # Arguments
/// * `graph` - The graph being analyzed
/// * `partition` - The partition to evaluate
/// 
/// # Returns
/// The modularity value of the partition
pub fn calculate_modularity(graph: &Graph, partition: &Partition) -> f64 {
    if graph.num_edges() == 0 {
        return 0.0;
    }

    let m = graph.num_edges() as f64;
    let mut modularity = 0.0;

    for &node_i in graph.nodes_iter() {
        for &node_j in graph.nodes_iter() {
            if node_i >= node_j {
                continue; // Avoid double counting
            }

            let same_community = match (partition.get(&node_i), partition.get(&node_j)) {
                (Some(&comm_i), Some(&comm_j)) => comm_i == comm_j && comm_i != -1,
                _ => false,
            };

            if same_community {
                let a_ij = if graph.has_edge(node_i, node_j) { 1.0 } else { 0.0 };
                let k_i = graph.degree(&node_i) as f64;
                let k_j = graph.degree(&node_j) as f64;
                let expected = (k_i * k_j) / (2.0 * m);
                
                modularity += a_ij - expected;
            }
        }
    }

    modularity / m
}

/// Validates that a partition is consistent with the graph
/// 
/// Checks that all nodes in the partition exist in the graph and vice versa.
/// 
/// # Arguments
/// * `graph` - The graph to validate against
/// * `partition` - The partition to validate
/// 
/// # Returns
/// True if the partition is valid, false otherwise
pub fn validate_partition(graph: &Graph, partition: &Partition) -> bool {
    // Check that all graph nodes are in the partition
    for node in graph.nodes_iter() {
        if !partition.contains_key(node) {
            return false;
        }
    }

    // Check that all partition nodes exist in the graph
    for &node in partition.keys() {
        if !graph.nodes.contains(&node) {
            return false;
        }
    }

    true
}

/// Gets the community assignment for a node
/// 
/// # Arguments
/// * `partition` - The partition to query
/// * `node` - The node to get community for
/// 
/// # Returns
/// The community ID if the node exists, otherwise None
pub fn get_node_community(partition: &Partition, node: NodeId) -> Option<CommunityId> {
    partition.get(&node).copied()
}

/// Gets all nodes belonging to a specific community
/// 
/// # Arguments
/// * `partition` - The partition to query
/// * `community` - The community ID to get nodes for
/// 
/// # Returns
/// Vector of node IDs in the specified community
pub fn get_community_nodes(partition: &Partition, community: CommunityId) -> Vec<NodeId> {
    partition
        .iter()
        .filter_map(|(&node, &comm)| {
            if comm == community {
                Some(node)
            } else {
                None
            }
        })
        .collect()
}

/// Counts the number of distinct communities in a partition
/// 
/// # Arguments
/// * `partition` - The partition to analyze
/// 
/// # Returns
/// Number of distinct communities (excluding isolated nodes with community -1)
pub fn count_communities(partition: &Partition) -> usize {
    let mut communities = partition.values().filter(|&&comm| comm != -1).collect::<Vec<_>>();
    communities.sort_unstable();
    communities.dedup();
    communities.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 0);
        graph.add_edge(4, 5);
        graph.finalize();
        graph
    }

    #[test]
    fn test_normalize_community_ids() {
        let graph = create_test_graph();
        let mut partition = Partition::default();
        partition.insert(0, 10);
        partition.insert(1, 10);
        partition.insert(2, 20);
        partition.insert(3, 20);
        partition.insert(4, 30);
        partition.insert(5, 30);

        let normalized = normalize_community_ids(&graph, partition);
        
        // Check that communities are now 0, 1, 2
        let communities: std::collections::HashSet<_> = normalized.values().copied().collect();
        let expected_communities: std::collections::HashSet<_> = [0, 1, 2].into_iter().collect();
        assert_eq!(communities, expected_communities);
    }

    #[test]
    fn test_isolated_nodes() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1);
        // Node 2 is isolated (no edges)
        graph.nodes.insert(2);
        graph.node_vec.push(2);
        graph.finalize();

        let mut partition = Partition::default();
        partition.insert(0, 1);
        partition.insert(1, 1);
        partition.insert(2, 1);

        let normalized = normalize_community_ids(&graph, partition);
        
        // Isolated node should have community -1
        assert_eq!(normalized[&2], -1);
        assert_eq!(normalized[&0], 0);
        assert_eq!(normalized[&1], 0);
    }

    #[test]
    fn test_validate_partition() {
        let graph = create_test_graph();
        let mut valid_partition = Partition::default();
        for node in graph.nodes_iter() {
            valid_partition.insert(*node, 0);
        }

        assert!(validate_partition(&graph, &valid_partition));

        // Invalid partition - missing node
        let mut invalid_partition = valid_partition.clone();
        invalid_partition.remove(&0);
        assert!(!validate_partition(&graph, &invalid_partition));

        // Invalid partition - extra node
        invalid_partition.insert(99, 0);
        assert!(!validate_partition(&graph, &invalid_partition));
    }

    #[test]
    fn test_get_node_community() {
        let mut partition = Partition::default();
        partition.insert(0, 1);
        partition.insert(1, 2);

        assert_eq!(get_node_community(&partition, 0), Some(1));
        assert_eq!(get_node_community(&partition, 1), Some(2));
        assert_eq!(get_node_community(&partition, 999), None);
    }

    #[test]
    fn test_get_community_nodes() {
        let mut partition = Partition::default();
        partition.insert(0, 1);
        partition.insert(1, 1);
        partition.insert(2, 2);

        let mut community_1_nodes = get_community_nodes(&partition, 1);
        community_1_nodes.sort();
        assert_eq!(community_1_nodes, vec![0, 1]);

        let community_2_nodes = get_community_nodes(&partition, 2);
        assert_eq!(community_2_nodes, vec![2]);

        let community_3_nodes = get_community_nodes(&partition, 3);
        assert!(community_3_nodes.is_empty());
    }

    #[test]
    fn test_count_communities() {
        let mut partition = Partition::default();
        partition.insert(0, 1);
        partition.insert(1, 1);
        partition.insert(2, 2);
        partition.insert(3, -1); // Isolated node

        assert_eq!(count_communities(&partition), 2);
    }
}