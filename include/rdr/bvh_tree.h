// You might wonder why we need this while having an existing bvh_accel
// The abstraction level of these two are different. bvh_tree as a generic class
// can be used to implement bvh_accel, but bvh_accel itself can only encapsulate
// TriangleMesh thus cannot be used with bvhtree.
#ifndef __BVH_TREE_H__
#define __BVH_TREE_H__

#include "math_aliases.h"
#include "rdr/accel.h"
#include "rdr/platform.h"
#include "rdr/primitive.h"
#include "rdr/ray.h"
#include <algorithm>
#include <cstdint>
#include <map>

RDR_NAMESPACE_BEGIN

template <typename DataType_>
class BVHNodeInterface {
public:
  using DataType = DataType_;

  // The only two required interfaces
  virtual AABB getAABB() const            = 0;
  virtual const DataType &getData() const = 0;

protected:
  // Interface spec
  BVHNodeInterface()  = default;
  ~BVHNodeInterface() = default;

  BVHNodeInterface(const BVHNodeInterface &)            = default;
  BVHNodeInterface &operator=(const BVHNodeInterface &) = default;
};

// TODO: check derived class's type
template <typename NodeType_>
class BVHTree final {
public:
  using NodeType  = NodeType_;
  using IndexType = int;

  // Context-local
  constexpr static int INVALID_INDEX = -1;
  constexpr static int CUTOFF_DEPTH  = 22;

  enum class EHeuristicProfile {
    EMedianHeuristic      = 0,  ///<! use centroid[depth%3]
    ESurfaceAreaHeuristic = 1,  ///<! use SAH (see PBRT)
  };

  // The actual node that represents the tree structure
  struct InternalNode {
    InternalNode() = default;
    InternalNode(IndexType span_left, IndexType span_right)
        : span_left(span_left), span_right(span_right) {}

    bool is_leaf{false};
    IndexType left_index{INVALID_INDEX};
    IndexType right_index{INVALID_INDEX};
    IndexType span_left{INVALID_INDEX};
    IndexType span_right{INVALID_INDEX};  // nodes[span_left, span_right)
    AABB aabb{};                          // The bounding box of the node
  };

  struct ParallelInternalNode {
    ParallelInternalNode() = default;
    ParallelInternalNode(IndexType left_index, IndexType right_index)
        : left_index(left_index), right_index(right_index) {}

    bool is_leaf{false};
    IndexType left_index{INVALID_INDEX};//internal node index - leaf node index +
    IndexType right_index{INVALID_INDEX};
    AABB aabb{};                          // The bounding box of the node
  };
  BVHTree()  = default;
  ~BVHTree() = default;

  /// General Interface
  size_t size() { return nodes.size(); }

  /// Nodes might be re-ordered
  void push_back(const NodeType &node) { nodes.push_back(node); }
  const AABB &getAABB() const { return internal_nodes[root_index].aabb; }

  /// reset build status
  void clear();

  /// *Can* be executed not only once
  void build();
  void build_parallel();

  template <typename Callback>
  bool intersect(Ray &ray, Callback callback) const {
    if (!is_built) return false;
    return intersect(ray, root_index, callback);
  }

private:
  EHeuristicProfile hprofile{EHeuristicProfile::EMedianHeuristic};

  bool is_built{false};
  IndexType root_index{INVALID_INDEX};

  vector<NodeType> nodes{};               /// The data nodes
  vector<InternalNode> internal_nodes{};  /// The internal nodes

  vector<IndexType> parallel_split_indices{};
  vector<IndexType> parallel_internal_indices{};
  vector<ParallelInternalNode> parallel_internal_nodes{};

  std::map<NodeType, uint32_t> morton_code_map{};
  /// Internal build
  IndexType build(
      int depth, const IndexType &span_left, const IndexType &span_right);
  
  // IndexType build_parallel(
      // const IndexType &span_left, const IndexType &span_right);
  
  void build_radix_tree();

  /// Internal intersect
  template <typename Callback>
  bool intersect(
      Ray &ray, const IndexType &node_index, Callback callback) const;
  
  void computeMortonCodes();

  int delta(int x, int y) const;
};

/* ===================================================================== *
 *
 * Implementation
 *
 * ===================================================================== */

template <typename _>
void BVHTree<_>::clear() {
  nodes.clear();
  internal_nodes.clear();
  is_built = false;
}

template <typename _>
void BVHTree<_>::build() {
  if (is_built) return;
  // pre-allocate memory
  internal_nodes.reserve(2 * nodes.size());
  root_index = build(0, 0, nodes.size());
  is_built   = true;
}

template <typename _>
typename BVHTree<_>::IndexType BVHTree<_>::build(
    int depth, const IndexType &span_left, const IndexType &span_right) {
  if (span_left >= span_right) return INVALID_INDEX;

  // early calculate bound
  AABB prebuilt_aabb;
  for (IndexType span_index = span_left; span_index < span_right; ++span_index)
    prebuilt_aabb.unionWith(nodes[span_index].getAABB());

  // TODO: setup the stop criteria
  //
  // You should fill in the stop criteria here.
  //
  // You may find the following variables useful:
  //
  // @see CUTOFF_DEPTH: The maximum depth you would like to build
  // @see span_left: The left index of the current span
  // @see span_right: The right index of the current span
  //
  
  if (depth >= CUTOFF_DEPTH || span_right - span_left == 1)
  {
    // create leaf node
    const auto &node = nodes[span_left];
    InternalNode result(span_left, span_right);
    result.is_leaf = true;
    result.aabb    = prebuilt_aabb;
    internal_nodes.push_back(result);
    return internal_nodes.size() - 1;
  }

  // You'll notice that the implementation here is different from the KD-Tree
  // ones, which re-use the node for both data-storing and organizing the real
  // tree structure. Here, for simplicity and generality, we use two different
  // types of nodes to ensure simplicity in interface, i.e. provided node does
  // not need to be aware of the tree structure.
  InternalNode result(span_left, span_right);

  // const int &dim = depth % 3; 
  const int &dim  = ArgMax(prebuilt_aabb.getExtent());
  IndexType count = span_right - span_left;
  IndexType split = INVALID_INDEX;

  if (hprofile == EHeuristicProfile::EMedianHeuristic) {
use_median_heuristic:
    split = span_left + count / 2;
    // Sort the nodes
    // after which, all centroids in [span_left, split) are LT than right
    // clang-format off

    // TODO: implement the median split here
    //
    // You should sort the nodes in [span_left, span_right) according to
    // their centroid's `dim`-th dimension, such that all nodes in
    // [span_left, split) are less than those in [split, span_right)
    //
    // You may find `std::nth_element` useful here.

    std::nth_element(nodes.begin() + span_left, nodes.begin() + split, nodes.begin() + span_right, [dim](const auto x, const auto y){ return x.getAABB().getCenter()[dim] < y.getAABB().getCenter()[dim];});
    

    // clang-format on
  } else if (hprofile == EHeuristicProfile::ESurfaceAreaHeuristic) {
use_surface_area_heuristic:
    // See
    // https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
    // for algorithm details. In general, like *decision tree*, we evaluate our
    // split with some measures.
    // Briefly speaking, "for a convex volume A *contained* in another larger
    // convex volume B , the conditional probability that a uniformly
    // distributed random ray passing through B will also pass through A is the
    // ratio of their surface areas"

    // TODO (BONUS): implement Surface area heuristic here
    //
    // You can then set @see BVHTree::hprofile to ESurfaceAreaHeuristic to
    // enable this feature.
    UNIMPLEMENTED;
  }

  // Build the left and right subtree
  result.left_index  = build(depth + 1, span_left, split);
  result.right_index = build(depth + 1, split, span_right);

  // Iterative merge
  result.aabb = prebuilt_aabb;

  internal_nodes.push_back(result);
  return internal_nodes.size() - 1;
}

template <typename _>
void BVHTree<_>::build_parallel() {
  if (is_built) return;
  computeMortonCodes();
  std::sort(nodes.begin(), nodes.end(), [&](const auto &a, const auto &b) {
    return morton_code_map[a] < morton_code_map[b];
  });
  // pre-allocate memory
  parallel_internal_nodes.reserve(nodes.size() - 1);
  build_radix_tree();

  internal_nodes.reserve(2 * nodes.size() - 1);

  // root_index = build_parallel(0, 0, nodes.size());
  is_built   = true;
}

// template <typename _>
// typename BVHTree<_>::IndexType BVHTree<_>::build_parallel(
//    const IndexType &span_left, const IndexType &span_right) {

// }
  
template <typename _>
void BVHTree<_>::build_radix_tree(){
  #pragma omp parallel for
  for (int i = 0; i < nodes.size() - 1; i++) {
    int d = Sign(
        delta(i, i + 1) - delta(i, i - 1));
    if (d == 0) d = 1;
    int delta_min = delta(i, i - d);
    int l_max = 2;
    while (delta(i, i + l_max * d) > delta_min) {
      l_max *= 2;
    }
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
      if (delta(i, i + (l + t) * d) > delta_min) {
        l += t;
      }
    }
    int j = i + l * d;
    int delta_node = delta(i, j);
    int s = 0;
    for (int t = l / 2; t >= 1; t /= 2) {
      if (delta(i, i + (s + t) * d) > delta_node) {
        s += t;
      }
    }
    int split = i + s * d + std::min(d, 0);
    int left, right;
    if (Min(i, j) == split) {
      left = split;
    }
    else {
      left = -split;
    }
    if (Max(i, j) == split + 1) {
      right = split + 1;
    }
    else {
      right = - (split + 1);
    }
    ParallelInternalNode pinode(left, right);
    parallel_internal_nodes.push_back(pinode);
  }
}
template <typename _>
void BVHTree<_>::computeMortonCodes() {
  Float xmax, ymax, zmax;
  Float xmin, ymin, zmin;
  for (int i = 0; i < nodes.size(); i++) 
  {
    const auto &aabb = nodes[i].getAABB();
    const auto centroid = aabb.getCenter();
    if (i == 0) 
    {
      xmin = xmax = centroid.x;
      ymin = ymax = centroid.y;
      zmin = zmax = centroid.z;
    } 
    else 
    {
      xmin = std::min(xmin, centroid.x);
      xmax = std::max(xmax, centroid.x);
      ymin = std::min(ymin, centroid.y);
      ymax = std::max(ymax, centroid.y);
      zmin = std::min(zmin, centroid.z);
      zmax = std::max(zmax, centroid.z);
    }
  }
  #pragma omp parallel for
  for (int i = 0; i < nodes.size(); i++) 
  {
    const auto &aabb = nodes[i].getAABB();
    const auto centroid = aabb.getCenter();
    // Map to [0, 1023]
    int x = static_cast<int>(
        1023 * (centroid.x - xmin) / (xmax - xmin));
    int y = static_cast<int>(
        1023 * (centroid.y - ymin) / (ymax - ymin));
    int z = static_cast<int>(
        1023 * (centroid.z - zmin) / (zmax - zmin));
    
    // Interleave bits
    int morton_code = 0;
    for (int j = 0; j < 10; j++) {
      morton_code |= ((x >> j) & 1) << (3 * j + 2);
      morton_code |= ((y >> j) & 1) << (3 * j + 1);
      morton_code |= ((z >> j) & 1) << (3 * j);
    }
    morton_code_map[nodes[i]] = morton_code;
  }
}

template <typename _>
int BVHTree<_>::delta(int x, int y) const {
  if (y > nodes.size() - 1 || y < 0) {
    return -1;
  }
  if (morton_code_map[nodes[x]] == morton_code_map[nodes[y]]) return 32;
  return __builtin_clz(morton_code_map[nodes[x]] ^ morton_code_map[nodes[y]]);
}

template <typename _>
template <typename Callback>
bool BVHTree<_>::intersect(
    Ray &ray, const IndexType &node_index, Callback callback) const {
  bool result      = false;
  const auto &node = internal_nodes[node_index];

  // Perform the actual pruning
  Float t_in, t_out;
  if (!node.aabb.intersect(ray, &t_in, &t_out)) return result;

  if (node.is_leaf) {
    for (IndexType span_index = node.span_left; span_index < node.span_right;
        ++span_index)
      result |= callback(ray, nodes[span_index].getData());
    return result;
  } else {
    // Recurse
    if (node.left_index != INVALID_INDEX)
      result |= intersect(ray, node.left_index, callback);
    if (node.right_index != INVALID_INDEX)
      result |= intersect(ray, node.right_index, callback);
    return result;
  }
}

RDR_NAMESPACE_END

#endif
