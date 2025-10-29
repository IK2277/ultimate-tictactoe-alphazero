#include "uttt_game.h"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace UTTT {

// デフォルトコンストラクタ
State::State() : active_board_(-1) {
    // すべてのセルを0で初期化
    for (auto& board : pieces_) {
        board.fill(0);
    }
    for (auto& board : enemy_pieces_) {
        board.fill(0);
    }
    main_board_pieces_.fill(0);
    main_board_enemy_pieces_.fill(0);
}

// カスタムコンストラクタ
State::State(const std::array<std::array<int, 9>, 9>& pieces,
             const std::array<std::array<int, 9>, 9>& enemy_pieces,
             const std::array<int, 9>& main_board_pieces,
             const std::array<int, 9>& main_board_enemy_pieces,
             int active_board)
    : pieces_(pieces)
    , enemy_pieces_(enemy_pieces)
    , main_board_pieces_(main_board_pieces)
    , main_board_enemy_pieces_(main_board_enemy_pieces)
    , active_board_(active_board) {
}

// 3並びチェック
bool State::check_win(const std::array<int, 9>& board_pieces) const {
    // 3並びかどうかチェックするヘルパー
    auto is_comp = [&](int x, int y, int dx, int dy) -> bool {
        for (int k = 0; k < 3; k++) {
            if (y < 0 || y > 2 || x < 0 || x > 2 || board_pieces[x + y * 3] == 0) {
                return false;
            }
            x += dx;
            y += dy;
        }
        return true;
    };

    // 斜め
    if (is_comp(0, 0, 1, 1) || is_comp(0, 2, 1, -1)) {
        return true;
    }
    
    // 縦横
    for (int i = 0; i < 3; i++) {
        if (is_comp(0, i, 1, 0) || is_comp(i, 0, 0, 1)) {
            return true;
        }
    }
    
    return false;
}

// 石の総数
int State::piece_count(const std::array<std::array<int, 9>, 9>& pieces_list) const {
    int count = 0;
    for (const auto& board : pieces_list) {
        for (int cell : board) {
            if (cell == 1) {
                count++;
            }
        }
    }
    return count;
}

// メインボードで相手が勝ったか
bool State::is_lose() const {
    return check_win(main_board_enemy_pieces_);
}

// 引き分けかどうか
bool State::is_draw() const {
    return !is_lose() && legal_actions().empty();
}

// ゲーム終了かどうか
bool State::is_done() const {
    return is_lose() || is_draw();
}

// 先手かどうか
bool State::is_first_player() const {
    return piece_count(pieces_) == piece_count(enemy_pieces_);
}

// 次の状態を取得
State State::next(int action) const {
    int board_idx = action / 9;
    int cell_idx = action % 9;
    
    // プレイヤーを交代し、盤面をコピー
    auto new_pieces = enemy_pieces_;
    auto new_enemy_pieces = pieces_;
    auto new_main_pieces = main_board_enemy_pieces_;
    auto new_main_enemy_pieces = main_board_pieces_;
    
    // 石を配置
    new_enemy_pieces[board_idx][cell_idx] = 1;
    
    // 小盤面で勝利したか判定
    bool small_board_win = check_win(new_enemy_pieces[board_idx]);
    
    if (small_board_win) {
        new_main_enemy_pieces[board_idx] = 1;
    } else {
        // 小盤面が引き分けか判定
        bool board_full = true;
        for (int j = 0; j < 9; j++) {
            if (new_pieces[board_idx][j] == 0 && new_enemy_pieces[board_idx][j] == 0) {
                board_full = false;
                break;
            }
        }
        if (board_full) {
            // 引き分けは両方の勝利としてマーク
            new_main_pieces[board_idx] = 1;
            new_main_enemy_pieces[board_idx] = 1;
        }
    }
    
    // 次のアクティブボードを決定
    int next_active_board = cell_idx;
    
    // その盤面が既に終了している場合は -1 (任意)
    bool target_board_finished = (new_main_pieces[next_active_board] == 1 ||
                                   new_main_enemy_pieces[next_active_board] == 1);
    
    if (target_board_finished) {
        next_active_board = -1;
    }
    
    return State(new_pieces, new_enemy_pieces,
                 new_main_pieces, new_main_enemy_pieces,
                 next_active_board);
}

// 合法手のリスト取得
std::vector<int> State::legal_actions() const {
    std::vector<int> actions;
    
    if (is_lose()) {
        return actions;
    }
    
    // プレイ可能な盤面のリスト作成
    std::vector<int> candidate_boards;
    
    if (active_board_ == -1) {
        // 任意: 終了していない全ての盤面が候補
        for (int i = 0; i < 9; i++) {
            if (main_board_pieces_[i] == 0 && main_board_enemy_pieces_[i] == 0) {
                candidate_boards.push_back(i);
            }
        }
    } else {
        // 強制: active_board が候補
        if (main_board_pieces_[active_board_] == 0 && 
            main_board_enemy_pieces_[active_board_] == 0) {
            candidate_boards.push_back(active_board_);
        } else {
            // 送り込まれた先が既に終了していた場合
            for (int i = 0; i < 9; i++) {
                if (main_board_pieces_[i] == 0 && main_board_enemy_pieces_[i] == 0) {
                    candidate_boards.push_back(i);
                }
            }
        }
    }
    
    // 候補の盤面から空いているセルを探す
    for (int board_idx : candidate_boards) {
        for (int cell_idx = 0; cell_idx < 9; cell_idx++) {
            if (pieces_[board_idx][cell_idx] == 0 && 
                enemy_pieces_[board_idx][cell_idx] == 0) {
                actions.push_back(board_idx * 9 + cell_idx);
            }
        }
    }
    
    return actions;
}

// 文字列表示
std::string State::to_string() const {
    const char* ox = is_first_player() ? "ox" : "xo";
    std::ostringstream oss;
    
    // 盤面表示
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 3; i++) {
                int board_idx = r * 3 + i;
                for (int j = 0; j < 3; j++) {
                    int cell_idx = c * 3 + j;
                    
                    char p = '-';
                    if (pieces_[board_idx][cell_idx] == 1) {
                        p = ox[0];
                    } else if (enemy_pieces_[board_idx][cell_idx] == 1) {
                        p = ox[1];
                    }
                    
                    oss << p << " ";
                }
                if (i < 2) oss << "| ";
            }
            oss << "\n";
        }
        if (r < 2) oss << "---------------------\n";
    }
    
    // メインボード状態
    oss << "\nMain Board Status:\n";
    for (int i = 0; i < 9; i++) {
        char mb_p = '.';
        if (main_board_pieces_[i] == 1 && main_board_enemy_pieces_[i] == 1) {
            mb_p = 'D'; // Draw
        } else if (main_board_pieces_[i] == 1) {
            mb_p = ox[0];
        } else if (main_board_enemy_pieces_[i] == 1) {
            mb_p = ox[1];
        }
        oss << mb_p;
        if (i % 3 == 2) oss << '\n';
    }
    
    oss << "Next Player: " << ox[0] << "\n";
    oss << "Active Board: " << (active_board_ == -1 ? "Any" : std::to_string(active_board_)) << "\n";
    
    return oss.str();
}

// 入力テンソル生成 (9x9x3) をフラット配列で返す
std::vector<float> State::to_input_tensor() const {
    std::vector<float> tensor(9 * 9 * 3, 0.0f);
    
    // 合法手を取得
    std::vector<int> legal_acts = legal_actions();
    
    // チャンネル 0: 自分の石
    // チャンネル 1: 相手の石
    // チャンネル 2: 合法手
    for (int board_idx = 0; board_idx < 9; board_idx++) {
        for (int cell_idx = 0; cell_idx < 9; cell_idx++) {
            int R = (board_idx / 3) * 3 + (cell_idx / 3);
            int C = (board_idx % 3) * 3 + (cell_idx % 3);
            
            // チャンネル 0 (自分)
            if (pieces_[board_idx][cell_idx] == 1) {
                tensor[R * 9 * 3 + C * 3 + 0] = 1.0f;
            }
            
            // チャンネル 1 (相手)
            if (enemy_pieces_[board_idx][cell_idx] == 1) {
                tensor[R * 9 * 3 + C * 3 + 1] = 1.0f;
            }
        }
    }
    
    // チャンネル 2 (合法手)
    for (int action : legal_acts) {
        int board_idx = action / 9;
        int cell_idx = action % 9;
        int R = (board_idx / 3) * 3 + (cell_idx / 3);
        int C = (board_idx % 3) * 3 + (cell_idx % 3);
        tensor[R * 9 * 3 + C * 3 + 2] = 1.0f;
    }
    
    return tensor;
}

} // namespace UTTT
