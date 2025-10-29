#pragma once

#include <vector>
#include <array>
#include <string>
#include <cstdint>

namespace UTTT {

// Ultimate Tic-Tac-Toe ゲーム状態
class State {
public:
    // 初期化
    State();
    
    // カスタムコンストラクタ
    State(const std::array<std::array<int, 9>, 9>& pieces,
          const std::array<std::array<int, 9>, 9>& enemy_pieces,
          const std::array<int, 9>& main_board_pieces,
          const std::array<int, 9>& main_board_enemy_pieces,
          int active_board);
    
    // ゲーム状態チェック
    bool is_lose() const;
    bool is_draw() const;
    bool is_done() const;
    bool is_first_player() const;
    
    // 次の状態を取得
    State next(int action) const;
    
    // 合法手のリスト取得
    std::vector<int> legal_actions() const;
    
    // デバッグ用文字列表示
    std::string to_string() const;
    
    // 入力テンソル生成 (9x9x3)
    std::vector<float> to_input_tensor() const;
    
    // メンバ変数へのアクセス（Python連携用）
    const std::array<std::array<int, 9>, 9>& get_pieces() const { return pieces_; }
    const std::array<std::array<int, 9>, 9>& get_enemy_pieces() const { return enemy_pieces_; }
    const std::array<int, 9>& get_main_board_pieces() const { return main_board_pieces_; }
    const std::array<int, 9>& get_main_board_enemy_pieces() const { return main_board_enemy_pieces_; }
    int get_active_board() const { return active_board_; }

private:
    // 盤面データ
    std::array<std::array<int, 9>, 9> pieces_; // [board_idx][cell_idx]
    std::array<std::array<int, 9>, 9> enemy_pieces_;
    std::array<int, 9> main_board_pieces_;
    std::array<int, 9> main_board_enemy_pieces_;
    int active_board_; // -1 = 任意, 0-8 = 指定された盤面
    
    // 内部ヘルパー関数
    bool check_win(const std::array<int, 9>& board_pieces) const;
    int piece_count(const std::array<std::array<int, 9>, 9>& pieces_list) const;
};

} // namespace UTTT
