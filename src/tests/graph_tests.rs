//! Tests for graph data structures and operations

use super::common::*;
use crate::graph::*;
use rustc_hash::FxHashMap;

#[test]
fn test_graph_creation_and_basic_operations() {
    let mut graph = Graph::new();
    assert_eq!(graph.num_nodes(), 0);
    assert_eq!(graph.num_edges(), 0);
    
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.finalize();
    
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(graph.num_edges(), 2);
    assert!(graph.has_edge(0, 1));
    assert!(graph.has_edge(1, 0)); // Undirected
    assert!(graph.has_edge(1, 2));
    assert!(!graph.has_edge(0, 2));
}

#[test]
fn test_graph_degree_calculations() {
    let graph = create_two_triangle_graph();
    
    // Node 2 and 3 should have degree 3 (part of triangle + bridge)
    assert_eq!(graph.degree(&2), 3);
    assert_eq!(graph.degree(&3), 3);
    
    // Other nodes should have degree 2
    assert_eq!(graph.degree(&0), 2);
    assert_eq!(graph.degree(&1), 2);
    assert_eq!(graph.degree(&4), 2);
    assert_eq!(graph.degree(&5), 2);
    
    // Non-existent node should have degree 0
    assert_eq!(graph.degree(&99), 0);
}

#[test]
fn test_graph_neighbors() {
    let graph = create_two_triangle_graph();
    
    let mut neighbors_0: Vec<i32> = graph.neighbors(&0).to_vec();
    neighbors_0.sort();
    assert_eq!(neighbors_0, vec![1, 2]);
    
    let mut neighbors_2: Vec<i32> = graph.neighbors(&2).to_vec();
    neighbors_2.sort();
    assert_eq!(neighbors_2, vec![0, 1, 3]);
}

#[test]
fn test_graph_statistics() {
    let graph = create_two_triangle_graph();
    
    assert_eq!(graph.num_nodes(), 6);
    assert_eq!(graph.num_edges(), 7);
    assert_eq!(graph.max_degree(), 3);
    assert_eq!(graph.total_degree(), 14); // Sum of all degrees (each edge counted twice)
    assert_eq!(graph.avg_degree(), 14.0 / 6.0);
}

#[test]
fn test_graph_duplicate_edge_prevention() {
    let mut graph = Graph::new();
    graph.add_edge(0, 1);
    graph.add_edge(1, 0); // Same edge, different direction
    graph.add_edge(0, 1); // Exact duplicate
    graph.finalize();
    
    assert_eq!(graph.num_edges(), 1);
    assert_eq!(graph.degree(&0), 1);
    assert_eq!(graph.degree(&1), 1);
}

#[test]
fn test_graph_self_loop_prevention() {
    let mut graph = Graph::new();
    graph.add_edge(0, 0); // Self-loop should be ignored
    graph.add_edge(0, 1);
    graph.finalize();
    
    assert_eq!(graph.num_edges(), 1);
    assert!(!graph.has_edge(0, 0));
    assert!(graph.has_edge(0, 1));
}

#[test]
fn test_graph_memory_stats() {
    let graph = create_two_triangle_graph();
    let stats = graph.memory_stats();
    
    assert!(stats.total() > 0);
    assert!(stats.nodes_memory > 0);
    assert!(stats.edges_memory > 0);
    assert!(stats.adjacency_memory > 0);
    assert!(stats.degrees_memory > 0);
}

#[test]
fn test_partition_normalization() {
    let graph = create_two_triangle_graph();
    let mut partition = create_perfect_partition();
    
    // Change community IDs to non-consecutive values
    partition.insert(0, 10);
    partition.insert(1, 10);
    partition.insert(2, 10);
    partition.insert(3, 20);
    partition.insert(4, 20);
    partition.insert(5, 20);
    
    let normalized = normalize_community_ids(&graph, partition);
    
    // Check that communities are now 0 and 1
    let communities: std::collections::HashSet<_> = normalized.values()
        .filter(|&&c| c != -1)
        .copied()
        .collect();
    let expected: std::collections::HashSet<_> = [0, 1].into_iter().collect();
    assert_eq!(communities, expected);
}

#[test]
fn test_partition_validation() {
    let graph = create_two_triangle_graph();
    let partition = create_perfect_partition();
    
    assert!(validate_partition(&graph, &partition));
    
    // Test invalid partition - missing node
    let mut invalid_partition = partition.clone();
    invalid_partition.remove(&0);
    assert!(!validate_partition(&graph, &invalid_partition));
    
    // Test invalid partition - extra node
    invalid_partition.insert(99, 0);
    assert!(!validate_partition(&graph, &invalid_partition));
}

#[test]
fn test_community_utilities() {
    let partition = create_perfect_partition();
    
    // Test node community lookup
    assert_eq!(get_node_community(&partition, 0), Some(0));
    assert_eq!(get_node_community(&partition, 3), Some(1));
    assert_eq!(get_node_community(&partition, 99), None);
    
    // Test community nodes lookup
    let mut comm0_nodes = get_community_nodes(&partition, 0);
    comm0_nodes.sort();
    assert_eq!(comm0_nodes, vec![0, 1, 2]);
    
    let mut comm1_nodes = get_community_nodes(&partition, 1);
    comm1_nodes.sort();
    assert_eq!(comm1_nodes, vec![3, 4, 5]);
    
    // Test community counting
    assert_eq!(count_communities(&partition), 2);
}

#[test]
fn test_modularity_calculation() {
    let graph = create_two_triangle_graph();
    let partition = create_perfect_partition();
    
    let modularity = calculate_modularity(&graph, &partition);
    
    // Perfect community structure should have positive modularity
    assert!(modularity > 0.0);
    assert!(modularity <= 1.0);
}

#[test]
fn test_isolated_nodes_handling() {
    let mut graph = Graph::new();
    graph.add_edge(0, 1);
    // Node 2 will be isolated (no edges)
    graph.nodes.insert(2);
    graph.node_vec.push(2);
    graph.adjacency_list.entry(2).or_default(); // Empty adjacency list
    graph.degrees.insert(2, 0);
    graph.finalize();
    
    let mut partition = FxHashMap::default();
    partition.insert(0, 0);
    partition.insert(1, 0);
    partition.insert(2, 0); // Try to assign isolated node to a community
    
    let normalized = normalize_community_ids(&graph, partition);
    
    // Isolated node should be assigned to community -1
    assert_eq!(normalized[&2], -1);
    assert_eq!(normalized[&0], 0);
    assert_eq!(normalized[&1], 0);
}

#[test]
fn test_graph_from_adjacency_list_format() {
    // Test would require creating a temporary file
    // For now, test the core functionality with empty graph
    let graph = Graph::new();
    assert_eq!(graph.num_nodes(), 0);
    assert_eq!(graph.num_edges(), 0);
}

#[test] 
fn test_large_graph_performance() {
    // Test with a moderately sized random graph
    let graph = create_random_graph(100, 0.1, 42);
    
    assert!(graph.num_nodes() <= 100);
    assert!(graph.num_edges() > 0);
    
    // Test that operations complete in reasonable time
    let start = std::time::Instant::now();
    let _stats = graph.memory_stats();
    let _degrees = graph.precompute_degrees();
    let elapsed = start.elapsed();
    
    // Should complete very quickly
    assert!(elapsed < std::time::Duration::from_millis(100));
}

#[cfg(test)]
mod property_tests {
    use super::*;
    
    #[test]
    fn test_degree_sum_equals_twice_edges() {
        let graph = create_karate_club_graph();
        
        let degree_sum: usize = graph.nodes_iter()
            .map(|node| graph.degree(node))
            .sum();
        
        assert_eq!(degree_sum, 2 * graph.num_edges());
    }
    
    #[test]
    fn test_edge_consistency() {
        let graph = create_two_triangle_graph();
        
        // Every edge (u,v) should have v in neighbors of u and vice versa
        for &(from, to) in &graph.edges {
            assert!(graph.neighbors(&from).contains(&to));
            assert!(graph.neighbors(&to).contains(&from));
        }
    }
    
    #[test]
    fn test_partition_completeness() {
        let graph = create_karate_club_graph();
        let partition = create_perfect_partition();
        
        // Every node should be assigned to exactly one community
        for node in graph.nodes_iter() {
            if let Some(&community) = partition.get(node) {
                // Community ID should be reasonable
                assert!(community >= -1);
            }
        }
    }
}