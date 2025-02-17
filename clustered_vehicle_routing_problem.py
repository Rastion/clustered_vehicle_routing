import os
import sys
import math
import random
from qubots.base_problem import BaseProblem

def read_elem(filename):
    with open(filename) as f:
        return [line.strip() for line in f.read().splitlines() if line.strip() != ""]

def compute_dist(x1, y1, x2, y2):
    return int(math.floor(math.sqrt((x1 - x2)**2 + (y1 - y2)**2) + 0.5))

def compute_distance_matrix(customers_x, customers_y):
    n = len(customers_x)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = compute_dist(customers_x[i], customers_y[i], customers_x[j], customers_y[j])
    return dist_matrix

def compute_distance_depot(depot_x, depot_y, customers_x, customers_y):
    n = len(customers_x)
    dist_depot = [0] * n
    for i in range(n):
        dist_depot[i] = compute_dist(depot_x, depot_y, customers_x[i], customers_y[i])
    return dist_depot

def read_input_cvrp(filename):
    """
    Reads a clustered VRP instance from a file following a TSPLib-like/GVRP format.
    Expected keywords:
      - DIMENSION: (total number of nodes, including depot)
      - VEHICLES: (number of trucks)
      - GVRP_SETS: (number of clusters)
      - CAPACITY: (truck capacity)
      - NODE_COORD_SECTION: coordinates for each node (node 1 is the depot; the remaining are customers)
      - GVRP_SET_SECTION: for each cluster, a list of node indices (ending with -1)
      - DEMAND_SECTION: for each cluster, its demand
    """
    tokens = read_elem(filename)
    it = iter(tokens)
    nb_nodes = 0
    nb_trucks = 0
    nb_clusters = 0
    truck_capacity = 0
    # Read header tokens until NODE_COORD_SECTION is found.
    while True:
        token = next(it)
        if token == "DIMENSION:":
            nb_nodes = int(next(it))
            nb_customers = nb_nodes - 1  # excluding depot
        elif token == "VEHICLES:":
            nb_trucks = int(next(it))
        elif token == "GVRP_SETS:":
            nb_clusters = int(next(it))
        elif token == "CAPACITY:":
            truck_capacity = int(next(it))
        elif token == "NODE_COORD_SECTION":
            break
    # Read node coordinates: node 1 is depot; nodes 2..nb_nodes are customers.
    depot_x = depot_y = 0
    customers_x = []
    customers_y = []
    for n in range(nb_nodes):
        node_id = int(next(it))
        x = float(next(it))
        y = float(next(it))
        if node_id == 1:
            depot_x, depot_y = x, y
        else:
            customers_x.append(x)
            customers_y.append(y)
    dist_matrix = compute_distance_matrix(customers_x, customers_y)
    dist_depot = compute_distance_depot(depot_x, depot_y, customers_x, customers_y)
    # Next token should be GVRP_SET_SECTION.
    token = next(it)
    if token != "GVRP_SET_SECTION":
        print("Expected token GVRP_SET_SECTION")
        sys.exit(1)
    clusters_data = [None] * nb_clusters
    for c in range(nb_clusters):
        cluster_id = int(next(it))
        cluster = []
        val = int(next(it))
        while val != -1:
            # Original customer indices are in 2..nb_nodes; convert to 0-indexed.
            cluster.append(val - 2)
            val = int(next(it))
        clusters_data[c] = cluster
    token = next(it)
    if token != "DEMAND_SECTION":
        print("Expected token DEMAND_SECTION")
        sys.exit(1)
    demands = [None] * nb_clusters
    for c in range(nb_clusters):
        cluster_id = int(next(it))
        demands[c] = int(next(it))
    return nb_customers, nb_trucks, nb_clusters, truck_capacity, dist_matrix, dist_depot, demands, clusters_data

class ClusteredVehicleRoutingProblem(BaseProblem):
    """
    Clustered Vehicle Routing Problem (cluVRP)

    A fleet of vehicles with uniform capacity must service clusters of customers.
    Each cluster (a group of nearby customers) must be visited entirely by a single vehicle.
    The instance data includes:
      - nb_customers: number of customers (excluding the depot)
      - nb_trucks: number of vehicles
      - nb_clusters: number of clusters (each cluster is a list of customer indices)
      - truck_capacity: maximum capacity per truck
      - dist_matrix: 2D distance matrix among customers (0-indexed)
      - dist_depot: distance from depot to each customer (0-indexed)
      - demands: an array of demands for each cluster
      - clusters_data: a list of clusters (each is a list of customer indices)

    Candidate Solution:
      A dictionary with two keys:
        - "truckSequences": a list (length nb_trucks) where each element is a list
           (sequence) of cluster indices (0-indexed). These lists form a partition of {0,...,nb_clusters-1}.
        - "clustersSequences": a list (length nb_clusters) where each element is a permutation
           (list) of indices from 0 to (number of customers in that cluster - 1), representing the order
           in which customers within the cluster are visited.

    Objective:
      For each cluster, compute the intra-cluster distance (using dist_matrix) from the ordered sequence,
      and record the first and last customer (from clustersSequences). Then, for each truck route (from truckSequences),
      compute the route distance as:
         depot to first cluster’s first customer + intra-cluster distances + inter-cluster distances +
         last cluster’s last customer to depot.
      Also, the total demand of clusters in each route must not exceed truck_capacity.
      The overall objective is to minimize the total distance traveled over all trucks.
    """
    def __init__(self, instance_file=None, nb_customers=None, nb_trucks=None, nb_clusters=None,
                 truck_capacity=None, dist_matrix=None, dist_depot=None, demands=None,
                 clusters_data=None):
        if instance_file is not None:
            self._load_instance(instance_file)
        else:
            if (nb_customers is None or nb_trucks is None or nb_clusters is None or
                truck_capacity is None or dist_matrix is None or dist_depot is None or
                demands is None or clusters_data is None):
                raise ValueError("Either instance_file or all instance parameters must be provided.")
            self.nb_customers = nb_customers
            self.nb_trucks = nb_trucks
            self.nb_clusters = nb_clusters
            self.truck_capacity = truck_capacity
            self.dist_matrix = dist_matrix
            self.dist_depot = dist_depot
            self.demands = demands
            self.clusters_data = clusters_data

    def _load_instance(self, filename):
        (self.nb_customers, self.nb_trucks, self.nb_clusters, self.truck_capacity,
         self.dist_matrix, self.dist_depot, self.demands, self.clusters_data) = read_input_cvrp(filename)

    def evaluate_solution(self, solution) -> float:
        """
        Evaluate a candidate solution.

        The candidate solution is a dictionary with keys:
          - "truckSequences": list of lists (one per vehicle) of cluster indices.
          - "clustersSequences": list of lists (one per cluster) of indices representing the order
             of customers within that cluster.
        
        For each cluster k, we compute:
          - intra_cluster_distance[k]: the sum of distances between consecutive customers (using dist_matrix)
            according to the order in clustersSequences[k].
          - initial_node[k]: the customer corresponding to the first index in clustersSequences[k].
          - end_node[k]: the customer corresponding to the last index in clustersSequences[k].
        
        For each truck route, we verify that the total demand (sum of demands for clusters visited)
        does not exceed truck_capacity and compute the route distance as:
          depot to initial_node of first cluster + intra-cluster distance for first cluster +
          for each subsequent cluster, add distance from end_node of previous cluster to initial_node of current cluster +
          intra-cluster distance for current cluster +
          finally, distance from end_node of last cluster to depot.
        
        The overall objective is the total distance over all trucks. If any capacity constraint is violated
        or if the candidate solution is invalid, return a high penalty.
        """
        PENALTY = 1e9
        if not isinstance(solution, dict):
            return PENALTY
        if "truckSequences" not in solution or "clustersSequences" not in solution:
            return PENALTY
        truckSeq = solution["truckSequences"]
        clustersSeq = solution["clustersSequences"]
        if not (isinstance(truckSeq, list) and isinstance(clustersSeq, list)):
            return PENALTY
        if len(truckSeq) != self.nb_trucks or len(clustersSeq) != self.nb_clusters:
            return PENALTY

        # Verify that every cluster appears exactly once among truckSequences.
        all_clusters = []
        for route in truckSeq:
            if not isinstance(route, list):
                return PENALTY
            all_clusters.extend(route)
        if sorted(all_clusters) != list(range(self.nb_clusters)):
            return PENALTY

        # For each cluster, compute intra-cluster distance, initial node, and end node.
        intra_cluster_distance = [0] * self.nb_clusters
        initial_node = [None] * self.nb_clusters
        end_node = [None] * self.nb_clusters
        for k in range(self.nb_clusters):
            seq = clustersSeq[k]
            # Validate that seq is a permutation of 0...m-1, where m = len(clusters_data[k])
            m = len(self.clusters_data[k])
            if sorted(seq) != list(range(m)):
                return PENALTY
            ordered_customers = [self.clusters_data[k][i] for i in seq]
            if not ordered_customers:
                return PENALTY
            initial_node[k] = ordered_customers[0]
            end_node[k] = ordered_customers[-1]
            dist_sum = 0
            for i in range(1, len(ordered_customers)):
                u = ordered_customers[i - 1]
                v = ordered_customers[i]
                dist_sum += self.dist_matrix[u][v]
            intra_cluster_distance[k] = dist_sum

        total_distance = 0
        # Evaluate each truck's route.
        for route in truckSeq:
            if not route:  # Empty route → distance = 0
                continue
            # Check capacity: total demand of clusters in route must be <= truck_capacity.
            route_demand = sum(self.demands[k] for k in route)
            if route_demand > self.truck_capacity:
                return PENALTY
            # Compute route distance.
            rdist = 0
            # From depot to the initial node of the first cluster.
            rdist += self.dist_depot[ initial_node[route[0]] ]
            # Add intra-cluster distance for first cluster.
            rdist += intra_cluster_distance[route[0]]
            # For each subsequent cluster in the route.
            for i in range(1, len(route)):
                prev = route[i - 1]
                curr = route[i]
                rdist += self.dist_matrix[ end_node[prev] ][ initial_node[curr] ]
                rdist += intra_cluster_distance[curr]
            # From the last cluster's end node back to the depot.
            rdist += self.dist_depot[ end_node[route[-1]] ]
            total_distance += rdist

        return total_distance

    def random_solution(self):
        """
        Generate a random candidate solution.

        - truckSequences: randomly partition the set {0, ..., nb_clusters - 1} among nb_trucks,
          then randomly shuffle the order within each truck.
        - clustersSequences: for each cluster, generate a random permutation of indices 0..(m-1),
          where m is the number of customers in that cluster.
        """
        clusters_list = list(range(self.nb_clusters))
        random.shuffle(clusters_list)
        truckSeq = [[] for _ in range(self.nb_trucks)]
        for idx, cluster in enumerate(clusters_list):
            truckSeq[idx % self.nb_trucks].append(cluster)
        for route in truckSeq:
            random.shuffle(route)
        clustersSeq = []
        for k in range(self.nb_clusters):
            m = len(self.clusters_data[k])
            perm = list(range(m))
            random.shuffle(perm)
            clustersSeq.append(perm)
        return {"truckSequences": truckSeq, "clustersSequences": clustersSeq}
