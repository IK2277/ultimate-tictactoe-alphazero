#pragma once

#include "uttt_game.h"
#include <vector>
#include <memory>
#include <functional>

namespace UTTT {

// ニューラルネットワークの推論結果
struct InferenceResult {
    std::vector<float> policy; // サイズ81の方策
    float value;                // 価値
};

// ニューラルネットワークの推論関数型
// 入力: 複数の状態 (バッチ)
// 出力: 各状態に対する (policy, value) のペア
using InferenceFunc = std::function<std::vector<InferenceResult>(const std::vector<State>&)>;

// MCTSノード
class Node {
public:
    Node(const State& state, float p);
    
    // リーフノードを探索
    std::pair<Node*, float> search_leaf(std::vector<Node*>& path);
    
    // ノードを展開
    void expand(const std::vector<float>& policies);
    
    // 価値を伝播
    void backpropagate(const std::vector<Node*>& path, float value);
    
    // 次の子ノードを選択 (PUCT最大)
    Node* next_child_node();
    
    // アクセサ
    const State& get_state() const { return state_; }
    int get_n() const { return n_; }
    float get_w() const { return w_; }
    const std::vector<std::unique_ptr<Node>>& get_children() const { return child_nodes_; }
    
private:
    State state_;
    float p_; // 方策確率
    float w_; // 累計価値
    int n_;   // 試行回数
    std::vector<std::unique_ptr<Node>> child_nodes_;
};

// MCTS探索の実行
// model: ニューラルネットワークの推論関数
// state: 現在の状態
// temperature: ボルツマン分布の温度 (0=最大値選択, >0=確率的)
// evaluate_count: シミュレーション回数
// batch_size: バッチ推論のサイズ
// 戻り値: 各合法手のスコア (確率分布)
std::vector<float> pv_mcts_scores(
    InferenceFunc model,
    const State& state,
    float temperature,
    int evaluate_count = 50,
    int batch_size = 8
);

// ボルツマン分布の適用
std::vector<float> boltzman(const std::vector<float>& xs, float temperature);

} // namespace UTTT
