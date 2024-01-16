import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;

class Pair {
    int value;
    int dist;

    Pair(int value, int dist) {
        this.value = value;
        this.dist = dist;
    }
}

public class Graph {
    static FastReader sc;

    public static void main(String[] args) {
        sc = new FastReader();
        Graph graph = new Graph();
        System.out.println("Enter number of nodes");
        int nodes = sc.ni();
        System.out.println("Enter number of edges");
        int edges = sc.ni();
        int[][] adjMatrix = graph.generateAdjacencyMatrix(nodes, edges);
        List<List<Integer>> adjList = graph.generateAdjacencyList(nodes, edges);
        List<List<List<Integer>>> costAdjList = graph.generateCostAdjacencyList(nodes, edges);
        List<Integer> bfsOfGraph = graph.bfsOfGraph(nodes, adjList);
        List<Integer> dfsOfGraph = graph.dfsOfGraph(nodes, adjList);
        boolean detectCycleUndirectedGraphUsingBFS = graph.detectCycleUndirectedGraphUsingBFS(nodes, adjList);
        boolean detectCycleUndirectedGraphUsingDFS = graph.detectCycleUndirectedGraphUsingDFS(nodes, adjList);
        // See application and Ques like No of Islands, Rotten Oranges, No of Distinct islands (set with BaseRow and BaseCol
        // concept), Bipartite(1-color(0,1) coloring) all based  on above Detection and BFS, DFS

        boolean detectCycleDirectedGraphUsingDFS = graph.detectCycleDirectedGraphUsingDFS(nodes, adjList);
        // For Toposort AsmstWeightption is DAG => Dirceted Acyclic Graph
        int[] topoSortUsingDFS = graph.topoSortUsingDFS(nodes, adjList);
        int[] topoSortUsingBFSKahnsAlgo = graph.topoSortUsingBFSKahnsAlgo(nodes, adjList);

        // Detect a cycle in Directed Graph using BFS is get its Topo Sort and if its size its not equal to nodes there is a cycle
        // otherwise not a cycle

        //Eventual SafeNodes ques using topoSort => App 1 - Using DFS if you have vis all paths of a node and end at no further node
        // return true and start storing as safe node, if it forms a cycle => currPathVis add leave it and dont add to safe node list
        // Using Kahn's reverse the adjList and start with indegree 0 , what all covred in toposort are safe

        int[] dist = graph.shortestPathInADAG(costAdjList, /*src*/ 0);

        int[] cost = graph.djikstra(costAdjList, 0);
        // Print Shortest path instead of getting - what you can do is just use memoization to store the
        // parent from where you were getting shortest path

    }

    private int[][] generateAdjacencyMatrix(int nodes, int edges) {

        int[][] adjacencyMatrix = new int[nodes][nodes];
        for (int i = 0; i < edges; i++) {
            int src = sc.ni(), dest = sc.ni();
            adjacencyMatrix[src][dest] = 1;
            adjacencyMatrix[dest][src] = 1;
        }
        return adjacencyMatrix;
    }

    private List<List<Integer>> generateAdjacencyList(int nodes, int edges) {
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < nodes; i++) {
            adjList.add(new ArrayList<>());
        }

        for (int i = 0; i < edges; i++) {
            int src = sc.ni(), dest = sc.ni();
            adjList.get(src).add(dest);
            adjList.get(dest).add(src);  // Remove if directed
        }
        return adjList;
    }

    private List<List<List<Integer>>> generateCostAdjacencyList(int nodes, int edges) {
        List<List<List<Integer>>> costAdjList = new ArrayList<>();

        for (int i = 0; i < nodes; i++) {
            costAdjList.add(new ArrayList<>());
        }

        for (int i = 0; i < edges; i++) {
            int src = sc.ni(), dest = sc.ni(), weight = sc.ni();
            costAdjList.get(src).add(new ArrayList<>(Arrays.asList(dest, weight)));
        }
        return costAdjList;
    }

    private List<Integer> bfsOfGraph(int nodes, List<List<Integer>> adjList) {
        // SC => O(N) TC => O(N) + O(2E)
        List<Integer> bfs = new ArrayList<>();
        Queue<Integer> q = new LinkedList<>();
        boolean[] vis = new boolean[nodes];
        q.add(0);
        vis[0] = true;

        while (!q.isEmpty()) {
            Integer node = q.poll();
            bfs.add(node);
            for (Integer neighbour : adjList.get(node)) {
                if (!vis[neighbour]) {
                    vis[neighbour] = true;
                    q.add(neighbour);
                }
            }
        }
        return bfs;
    }

    private List<Integer> dfsOfGraph(int nodes, List<List<Integer>> adjList) {
        // SC => O(N) TC => O(N) + O(2E)
        boolean[] vis = new boolean[nodes];
        List<Integer> dfs = new ArrayList<>();
        dfs(0, vis, dfs, adjList);
        return dfs;
    }

    private void dfs(int node, boolean[] vis, List<Integer> dfs, List<List<Integer>> adjList) {
        vis[node] = true;
        dfs.add(node);
        for (Integer neighbour : adjList.get(node)) {
            if (!vis[neighbour]) {
                dfs(neighbour, vis, dfs, adjList);
            }
        }
    }

    private boolean detectCycleUndirectedGraphUsingBFS(int nodes, List<List<Integer>> adjList) {
        boolean[] vis = new boolean[nodes];
        for (int i = 0; i < nodes; i++) {
            if (!vis[i]) {
                if (detectCycleUndirectedGraphUsingBFS(adjList, vis, /*src
                 */ i, /*parent*/ -1)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean detectCycleUndirectedGraphUsingBFS(List<List<Integer>> adjList, boolean[] vis, int src, int parent) {
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{src, parent});
        vis[src] = true;
        while (!q.isEmpty()) {
            int[] p = q.poll();
            for (int neighbour : adjList.get(p[0])) {
                if (!vis[neighbour]) {
                    q.offer(new int[]{neighbour, p[0]});
                } else if (neighbour != p[1]) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean detectCycleUndirectedGraphUsingDFS(int nodes, List<List<Integer>> adjList) {
        boolean[] vis = new boolean[nodes];
        for (int i = 0; i < nodes; i++) {
            if (!vis[i]) {
                if (detectCycleUndirectedGraphUsingDFS(adjList, vis, /*src
                 */ i, /*parent*/ -1)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean detectCycleUndirectedGraphUsingDFS(List<List<Integer>> adjList, boolean[] vis, int src, int parent) {
        vis[src] = true;
        for (Integer neighbour : adjList.get(src)) {
            if (!vis[neighbour]) {
                if (detectCycleUndirectedGraphUsingDFS(adjList, vis, neighbour, src)) {
                    return true;
                }
            } else if (neighbour != parent) {
                return true;
            }
        }
        return false;
    }

    private boolean detectCycleDirectedGraphUsingDFS(int nodes, List<List<Integer>> adjList) {
        boolean[] vis = new boolean[nodes];
        boolean[] currPathVis = new boolean[nodes]; // Extra as compared to directed coz we have to keep
        // Track of the current path coz there can be two diff path meeting on same node
        // that is not cyclic

        for (int i = 0; i < nodes; i++) {
            if (!vis[i]) {
                if (detectCycleDirectedGraphUsingDFS(vis, currPathVis, adjList, i)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean detectCycleDirectedGraphUsingDFS(boolean[] vis, boolean[] currPathVis, List<List<Integer>> adjList, int src) {
        vis[src] = true;
        currPathVis[src] = true;
        for (Integer neighbour : adjList.get(src)) {
            if (!vis[neighbour]) {
                if (detectCycleDirectedGraphUsingDFS(vis, currPathVis, adjList, neighbour)) {
                    return true;
                }
            } else if (currPathVis[neighbour]) {
                return true;
            }
        }
        currPathVis[src] = false;
        return false;
    }

    private int[] topoSortUsingDFS(int nodes, List<List<Integer>> adjList) {
        boolean[] vis = new boolean[nodes];
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < nodes; i++) {
            if (!vis[i]) {
                dfs(adjList, vis, st, i);
            }
        }
        int[] topoSort = new int[nodes];
        int c = 0;
        while (!st.isEmpty()) {
            topoSort[c++] = st.pop();
        }
        return topoSort;
    }

    private void dfs(List<List<Integer>> adjList, boolean[] vis, Stack<Integer> st, int node) {
        vis[node] = true;
        for (Integer neighbour : adjList.get(node)) {
            if (!vis[neighbour]) {
                dfs(adjList, vis, st, neighbour);
            }
        }
        st.push(node);
    }

    private int[] topoSortUsingBFSKahnsAlgo(int nodes, List<List<Integer>> adjList) {
        int[] indegree = new int[nodes];
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < nodes; i++) {
            for (Integer neighbour : adjList.get(i)) {
                indegree[neighbour]++;
            }
        }
        for (int i = 0; i < nodes; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
            }
        }
        int c = 0;
        int[] topoSort = new int[nodes];
        while (!q.isEmpty()) {
            topoSort[c++] = q.peek();
            for (Integer neighbour : adjList.get(q.poll())) {
                indegree[neighbour]--;
                if (indegree[neighbour] == 0) {
                    q.offer(neighbour);
                }
            }
        }
        return topoSort;
    }

    private int[] shortestPathInADAG(List<List<List<Integer>>> costAdjList, int src) {
        Stack<Integer> topoSort = new Stack<>();
        int N = costAdjList.size();
        boolean[] vis = new boolean[N];
        getTopoOnWeightedList(costAdjList, vis, topoSort, src);

        // calc distance from toposort
        int[] dist = new int[N];
        Arrays.fill(dist, (int) 1e9);
        dist[0] = 0;
        while (!topoSort.isEmpty()) {
            int curr = topoSort.pop();
            for (List<Integer> neighbour : costAdjList.get(curr)) {
                if (dist[curr] + neighbour.get(1) < dist[neighbour.get(0)]) {
                    dist[neighbour.get(0)] = dist[curr] + neighbour.get(1);
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (dist[i] == (int) 1e9) {
                dist[i] = -1;
            }
        }

        return dist;
    }

    private void getTopoOnWeightedList(List<List<List<Integer>>> costAdjList, boolean[] vis, Stack<Integer> topoSort, int src) {
        vis[src] = true;
        for (List<Integer> neighbour : costAdjList.get(src)) {
            if (!vis[neighbour.get(0)]) {
                getTopoOnWeightedList(costAdjList, vis, topoSort, neighbour.get(0));
            }
        }
    }

    private int[] djikstra(List<List<List<Integer>>> adjList, int src) {

        // Applicable for graphs with non-negative wt.
        // TC -> E * logV => E => No of Edges, V => No of Nodes

        PriorityQueue<Pair> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.dist));

        int[] shortestDistanceFromSource = new int[adjList.size()];
        Arrays.fill(shortestDistanceFromSource, Integer.MAX_VALUE);

        shortestDistanceFromSource[src] = 0;
        pq.add(new Pair(src, 0));

        while (!pq.isEmpty()) {
            Pair p = pq.peek();
            int dist = p.dist;
            int node = p.value;
            pq.remove();
            for (List<Integer> neighbour : adjList.get(node)) {
                if (shortestDistanceFromSource[neighbour.get(0)] > dist + neighbour.get(1)) {
                    shortestDistanceFromSource[neighbour.get(0)] = dist + neighbour.get(1);
                    pq.add(new Pair(neighbour.get(0), shortestDistanceFromSource[neighbour.get(0)]));
                }
            }
        }
        return shortestDistanceFromSource;
    }

    public int CheapestFLight(int nodes, int[][] edges, int src, int dst, int k) {
        // Code here
        // Instead of PQ it can be a normal Queue based on stops
        // And final conditon can be it should be dest on k+1 th stop
        // Otherwise don't put it in Queue
        List<List<List<Integer>>> costAdjList = generateCostAdjacencyList(nodes, edges);
        Queue<int[]> q = new LinkedList<>();
        int[] shortestDistanceFromSource = new int[costAdjList.size()];
        Arrays.fill(shortestDistanceFromSource, Integer.MAX_VALUE);

        shortestDistanceFromSource[src] = 0;
        // {stops, src, dist}
        q.offer(new int[]{0, src, 0});
        while (!q.isEmpty()) {
            int[] curr = q.poll();
            for (List<Integer> neighbour : costAdjList.get(curr[1])) {
                if (shortestDistanceFromSource[neighbour.get(0)] > curr[2] + neighbour.get(1)) {
                    if (curr[0] != k) {
                        shortestDistanceFromSource[neighbour.get(0)] = curr[2] + neighbour.get(1);
                        q.add(new int[]{curr[0] + 1, neighbour.get(0), shortestDistanceFromSource[neighbour.get(0)]});
                    } else {
                        if (neighbour.get(0) == dst) {
                            shortestDistanceFromSource[neighbour.get(0)] = curr[2] + neighbour.get(1);
                        }
                    }
                }
            }
        }

        return shortestDistanceFromSource[dst] != Integer.MAX_VALUE ? shortestDistanceFromSource[dst] : -1;
    }

    private List<List<List<Integer>>> generateCostAdjacencyList(int nodes, int[][] edges) {
        List<List<List<Integer>>> costAdjList = new ArrayList<>();

        for (int i = 0; i < nodes; i++) {
            costAdjList.add(new ArrayList<>());
        }

        for (int i = 0; i < edges.length; i++) {
            int src = edges[i][0], dest = edges[i][1], weight = edges[i][2];
            costAdjList.get(src).add(new ArrayList<>(Arrays.asList(dest, weight)));
        }
        return costAdjList;
    }

    private int[] bellman_ford(int nodes, ArrayList<ArrayList<Integer>> edges, int src) {
        // Write your code here
        // Relax the edges Nodes-1 time asmstWeighte skew tree
        // For Negative cycle try relaxing one more time and if waits are still reducing it definitely contains
        // negative cycle
        // TC => nodes * Edges
        // GFG -> https://www.geeksforgeeks.org/problems/distance-from-the-source-bellman-ford-algorithm/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=distance-from-the-source-bellman-ford-algorithm
        int[] shortestDistanceFromSource = new int[nodes];
        Arrays.fill(shortestDistanceFromSource, (int) 1e8);

        shortestDistanceFromSource[src] = 0;

        for (int i = 0; i < nodes; i++) {
            for (ArrayList<Integer> edge : edges) {
                if (shortestDistanceFromSource[edge.get(0)] != 1e8 && shortestDistanceFromSource[edge.get(0)] + edge.get(2) < shortestDistanceFromSource[edge.get(1)]) {
                    shortestDistanceFromSource[edge.get(1)] = shortestDistanceFromSource[edge.get(0)] + edge.get(2);
                }
            }
        }
        // For negative cycle try relaxing one more time
        for (ArrayList<Integer> edge : edges) {
            if (shortestDistanceFromSource[edge.get(0)] != 1e8 && shortestDistanceFromSource[edge.get(0)] + edge.get(2) < shortestDistanceFromSource[edge.get(1)]) {
                return new int[]{-1};
            }
        }
        return shortestDistanceFromSource;
    }

    private void floydWarshall(int[][] matrix) {
        // Code here
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == -1) {
                    matrix[i][j] = (int) 1e9;
                }
                if (i == j) {
                    matrix[i][j] = 0;
                }
            }
        }
        // TC => O(N^3)
        // After all iterations will have the shortest path from each node to every other node
        // For negative cycle we can check if matrix[i][i] < 0 => it implies -ve cycle
        for (int via = 0; via < n; via++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] = Math.min(matrix[i][j], matrix[i][via] + matrix[via][j]);
                }
            }
        }

        // Changing as -1 if not reachable
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == (int) 1e9) {
                    matrix[i][j] = -1;
                }
            }
        }
    }

    private int minimumSpanningTreePrims(int nodes, int edge, int[][] edges) {
        // Code Here.
        // Start from any node and push in PQ => {wt, node, parent} => for starting node we will be taking
        // as -1
        // Visit all adjacent nodes if !vis add them to PQ else continue
        // while taking out from PQ if parent != -1 mark this node as visited and add edge to MST
        List<List<List<Integer>>> costAdjList = generateCostAdjacencyList(nodes, edges);
        int mstWeight = 0;
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        pq.offer(new int[]{0, 0, -1});
        boolean[] vis = new boolean[nodes];
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            //            System.out.println(curr[0] + " " + curr[1] + " " + curr[2]); => Remember to adjListCode to undirected from directed
            if (!vis[curr[1]]) {
                for (List<Integer> neighbour : costAdjList.get(curr[1])) {
                    if (!vis[neighbour.get(0)]) {
                        pq.offer(new int[]{neighbour.get(1), neighbour.get(0), curr[1]});
                    }
                }
                vis[curr[1]] = true;
                mstWeight += curr[0];
                if (curr[2] != -1) {
                    res.add(new ArrayList<>(Arrays.asList(curr[2], curr[1])));
                }
            }
        }
        return mstWeight;
    }

    private int minimumSpanningTreeKruskals(int nodes, int edge, int[][] edges) {
        // ElogE to sort + E*4*alpha
        Arrays.sort(edges, (e1, e2) -> Integer.compare(e1[2], e2[2]));
        int mstWeight = 0;
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        DisjointSet ds = new DisjointSet(nodes);
        for (int i = 0; i < edges.length; i++) {
            if (ds.findParent(edges[i][0]) != ds.findParent(edges[i][1])) {
                ds.unionBySize(edges[i][0], edges[i][1]);
                mstWeight += edges[i][2];
                System.out.println(edges[i][0] + " " + edges[i][1] + " " + edges[i][2]);
                res.add(new ArrayList<>(Arrays.asList(edges[i][0], edges[i][1])));
            }
        }
        return mstWeight;
    }

    private List<Integer> numOfIslands(int rows, int cols, int[][] operators) {
        //Your code here
        DisjointSet ds = new DisjointSet(rows * cols);
        List<Integer> res = new ArrayList<>();
        boolean[][] vis = new boolean[rows][cols];
        int[] drow = {-1, 0, 1, 0};
        int[] dcol = {0, -1, 0, 1};
        int count = 0;
        for (int[] operate : operators) {
            if (vis[operate[0]][operate[1]]) {
                res.add(count);
                continue;
            }
            vis[operate[0]][operate[1]] = true;
            count++; // Addition of new Island then we will handle all new pieces
            // Now move in all four directions and if can be combined reduce the count by -1;
            int currNode = operate[0] * cols + operate[1];
            for (int i = 0; i < 4; i++) {
                int newRow = operate[0] + drow[i];
                int newCol = operate[1] + dcol[i];
                if(isValid(newRow, newCol, rows, cols)) {
                    if(vis[newRow][newCol]) {
                        int adjNode = newRow * cols + newCol;
                        if(ds.findParent(currNode) != ds.findParent(adjNode)) {
                            count--;
                            ds.unionByRank(currNode, adjNode);
                        }
                    }
                }
            }
            res.add(count);
        }
        return res;
    }

    private boolean isValid(int newRow, int newCol, int rows, int cols) {
        return newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols;
    }

    private int kosaraju(int nodes, List<List<Integer>> adjList)
    {
        // code here
        // Sort all the nodes in order of finishing time
        // Reverse the graph
        // Again do the dfs to get all Strongly Connected Componenets

        Stack<Integer> st = new Stack<>(); // Will store in order of finishing time
        boolean[] vis = new boolean[nodes];

        // Graph can also be disconeected
        for (int i = 0; i < nodes; i++) {
            if(!vis[i]) {
                dfs(adjList, vis, st, /*src*/ i);
            }
        }

        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        List<List<Integer>> revAdjList = new ArrayList<>();
        for (int i = 0; i < nodes; i++) {
            revAdjList.add(new ArrayList<>());
        }

        for (int i = 0; i < nodes; i++) {
            for (int x : adjList.get(i)) {
                revAdjList.get(x).add(i);
            }
        }

        Arrays.fill(vis, false);

        while(!st.isEmpty()) {
            int i = st.peek();
            st.pop();
            if(!vis[i]) {
                ArrayList<Integer> scc = new ArrayList<>();
                dfs(revAdjList, scc, vis, i);
                res.add(scc);
            }
        }

        return res.size();

    }

    private void dfs(List<List<Integer>> revAdjList, ArrayList<Integer> scc, boolean[] vis, int i) {
        vis[i] = true;
        for(int neighbour : revAdjList.get(i)) {
            if (!vis[neighbour]) {
                dfs(revAdjList, scc, vis, neighbour);
            }
        }
        scc.add(i);
    }

    // Bridges in Graph
    private List<List<Integer>> criticalConnections(int nodes, List<List<Integer>> edges) {
        // prepare Graph
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < nodes; i++) {
            adjList.add(new ArrayList<>());
        }

        for (int i = 0; i < edges.size(); i++) {
            int src = edges.get(i).get(0), dest = edges.get(i).get(1);
            adjList.get(src).add(dest);
            adjList.get(dest).add(src);  // Remove if directed
        }

        List<List<Integer>> bridges = new ArrayList<>();

        boolean[] vis = new boolean[nodes];
        int[] insertionTime = new int[nodes];
        int[] lowestTime = new int[nodes];
        int timer = 0;

        for(int i = 0; i < nodes; i++) {
            if(!vis[i]) {
                dfsBridge(adjList, vis, insertionTime, lowestTime, timer, i, /*parent*/ - 1, bridges);
            }
        }

        // for (int i = 0; i < nodes; i++) {
        //     System.out.println(i + "=>" + insertionTime[i] + "-" + lowestTime[i]);
        // }

        return bridges;
    }

    private void dfsBridge(List<List<Integer>> adjList, boolean[] vis, int[] insertionTime, int[] lowestTime, int timer, Integer node, Integer parent, List<List<Integer>> bridges) {
        vis[node] = true;
        insertionTime[node] = timer; // Time of Insertion using DFS
        lowestTime[node] = timer;  // lowest of self and adjacent apart from Parent
        for(int neighbour : adjList.get(node)) {
            if(neighbour == parent) {
                continue;
            } else if(!vis[neighbour]) {
                dfsBridge(adjList, vis, insertionTime, lowestTime, (timer + 1), neighbour, node, bridges);
                lowestTime[node] = Math.min(lowestTime[neighbour], lowestTime[node]);
                // If its greater than for sure this edge is a bridge
                if(lowestTime[neighbour] > insertionTime[node]) {  // => Remember Condition carefully
                    bridges.add(new ArrayList<>(Arrays.asList(node, neighbour)));
                }
            } else {
                // If its already visited we will take lower time from neighbour
                // coz its definitely not a bridge
                lowestTime[node] = Math.min(lowestTime[neighbour], lowestTime[node]);
            }
        }

    }
}

class DisjointSet {
    int[] rank;
    int[] parent;
    int[] size;

    DisjointSet(int n) {
        rank = new int[n];
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            rank[i] = 0;
            parent[i] = i;
            size[i] = 1;
        }
    }

    public int findParent(int node) {
        if (parent[node] == node) {
            return node;
        }
        return parent[node] = findParent(parent[node]);
    }

    public void unionByRank(int node1, int node2) {
        int par1 = findParent(node1);
        int par2 = findParent(node2);

        if (par1 == par2) {
            return;
        }
        if (rank[par1] < rank[par2]) {
            parent[par1] = par2;
        } else if (rank[par1] > rank[par2]) {
            parent[par2] = par1;
        } else {
            parent[par1] = par2;
            rank[par2]++;
        }
    }

    public void unionBySize(int node1, int node2) {
        int par1 = findParent(node1);
        int par2 = findParent(node2);
        if (par1 == par2) {
            return;
        }
        if (size[par1] < size[par2]) {
            parent[par1] = par2;
            size[par2] += size[par1];
        } else {
            parent[par2] = par1;
            size[par1] += size[par2];
        }
    }
}

