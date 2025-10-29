#include "uttt_mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace UTTT {

// ノードのコンストラクタ
Node::Node(const State& state, float p)
    : state_(state), p_(p), w_(0.0f), n_(0) {
}

// リーフノードを探索
std::pair<Node*, float> Node::search_leaf(std::vector<Node*>& path) {
    path.push_back(this);
    
    // ゲーム終了時
    if (state_.is_done()) {
        float value = state_.is_lose() ? -1.0f : 0.0f;
        return {this, -value}; // 相手から見た価値
    }
    
    // 子ノードが存在しない (リーフ)
    if (child_nodes_.empty()) {
        return {this, 0.0f}; // 価値は未評価
    }
    
    // 子ノードが存在する
    Node* next_node = next_child_node();
    return next_node->search_leaf(path);
}

// ノードを展開
void Node::expand(const std::vector<float>& policies) {
    std::vector<int> legal_acts = state_.legal_actions();
    
    for (size_t i = 0; i < legal_acts.size(); i++) {
        int action = legal_acts[i];
        float policy = (i < policies.size()) ? policies[i] : 0.0f;
        State next_state = state_.next(action);
        child_nodes_.push_back(std::make_unique<Node>(next_state, policy));
    }
}

// 価値を伝播
void Node::backpropagate(const std::vector<Node*>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node* node = *it;
        node->w_ += value;
        node->n_ += 1;
        value = -value; // 親ノードから見た価値に反転
    }
}

// 次の子ノードを選択 (PUCT最大)
Node* Node::next_child_node() {
    const float C_PUCT = 1.0f;
    
    int total_n = 0;
    for (const auto& child : child_nodes_) {
        total_n += child->n_;
    }
    
    float sqrt_total = std::sqrt(static_cast<float>(total_n));
    float max_pucb = -1e9f;
    Node* best_child = nullptr;
    
    for (const auto& child : child_nodes_) {
        float q = (child->n_ > 0) ? (-child->w_ / child->n_) : 0.0f;
        float u = C_PUCT * child->p_ * sqrt_total / (1 + child->n_);
        float pucb = q + u;
        
        if (pucb > max_pucb) {
            max_pucb = pucb;
            best_child = child.get();
        }
    }
    
    return best_child;
}

// MCTS探索の実行
std::vector<float> pv_mcts_scores(
    InferenceFunc model,
    const State& state,
    float temperature,
    int evaluate_count,
    int batch_size
) {
    // ルートノードの作成
    Node root_node(state, 0.0f);
    
    // ルートノードを即座に展開（初期方策は均等分布）
    std::vector<int> root_legal_actions = state.legal_actions();
    if (root_legal_actions.empty()) {
        return std::vector<float>(); // 合法手がない場合は空を返す
    }
    
    // 初期方策（均等分布）
    float uniform_policy = 1.0f / root_legal_actions.size();
    std::vector<float> initial_policies(root_legal_actions.size(), uniform_policy);
    root_node.expand(initial_policies);
    
    // バッチ評価用のキュー
    std::vector<Node*> leaves_to_eval;
    std::vector<std::vector<Node*>> paths_to_leaves;
    
    for (int i = 0; i < evaluate_count; i++) {
        // 1. リーフノードを探索
        std::vector<Node*> path;
        auto [leaf, value] = root_node.search_leaf(path);
        
        // 2. ゲーム終了ノードの場合
        if (leaf->get_state().is_done()) {
            leaf->backpropagate(path, value);
            continue;
        }
        
        // 3. 未評価のリーフノードの場合（n == 0）のみキューに追加
        if (leaf->get_n() == 0 && leaf->get_children().empty()) {
            leaves_to_eval.push_back(leaf);
            paths_to_leaves.push_back(path);
        }
        
        // 4. バッチサイズに達したか、最後のシミュレーション
        if (static_cast<int>(leaves_to_eval.size()) >= batch_size || i == evaluate_count - 1) {
            if (!leaves_to_eval.empty()) {
                // 5. バッチ推論を実行
                std::vector<State> states_batch;
                for (Node* leaf : leaves_to_eval) {
                    states_batch.push_back(leaf->get_state());
                }
                
                std::vector<InferenceResult> results_batch = model(states_batch);
                
                // 6. 結果を各リーフに配布
                for (size_t j = 0; j < leaves_to_eval.size(); j++) {
                    Node* leaf = leaves_to_eval[j];
                    const auto& path = paths_to_leaves[j];
                    const auto& result = results_batch[j];
                    
                    // 合法手のみの方策を抽出
                    std::vector<int> legal_acts = leaf->get_state().legal_actions();
                    std::vector<float> legal_policies;
                    float policy_sum = 0.0f;
                    
                    for (int action : legal_acts) {
                        float p = (action < static_cast<int>(result.policy.size())) ? result.policy[action] : 0.0f;
                        legal_policies.push_back(p);
                        policy_sum += p;
                    }
                    
                    // 正規化
                    if (policy_sum > 0) {
                        for (float& p : legal_policies) {
                            p /= policy_sum;
                        }
                    } else {
                        // すべて0の場合は均等分布
                        float uniform = legal_policies.empty() ? 0.0f : 1.0f / legal_policies.size();
                        std::fill(legal_policies.begin(), legal_policies.end(), uniform);
                    }
                    
                    leaf->expand(legal_policies);
                    leaf->backpropagate(path, result.value);
                }
                
                // キューをクリア
                leaves_to_eval.clear();
                paths_to_leaves.clear();
            }
        }
    }
    
    // 合法手の試行回数を取得
    std::vector<float> scores;
    for (const auto& child : root_node.get_children()) {
        scores.push_back(static_cast<float>(child->get_n()));
    }
    
    // 温度に応じた処理
    if (temperature == 0.0f) {
        // 最大値のみ1
        auto max_it = std::max_element(scores.begin(), scores.end());
        std::fill(scores.begin(), scores.end(), 0.0f);
        if (max_it != scores.end()) {
            *max_it = 1.0f;
        }
    } else {
        // ボルツマン分布
        scores = boltzman(scores, temperature);
    }
    
    return scores;
}

// ボルツマン分布
std::vector<float> boltzman(const std::vector<float>& xs, float temperature) {
    std::vector<float> result;
    float sum = 0.0f;
    
    for (float x : xs) {
        float val = std::pow(x, 1.0f / temperature);
        result.push_back(val);
        sum += val;
    }
    
    if (sum > 0) {
        for (float& val : result) {
            val /= sum;
        }
    }
    
    return result;
}

} // namespace UTTT
