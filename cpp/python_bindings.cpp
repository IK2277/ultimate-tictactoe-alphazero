#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "uttt_game.h"
#include "uttt_mcts.h"

namespace py = pybind11;

// Pythonの推論関数をC++の InferenceFunc にラップ
UTTT::InferenceFunc wrap_python_inference(py::object python_func) {
    return [python_func](const std::vector<UTTT::State>& states) -> std::vector<UTTT::InferenceResult> {
        // C++の状態リストをPythonリストに変換
        py::list py_states;
        for (const auto& state : states) {
            py_states.append(state);
        }
        
        // Python関数を呼び出し
        py::object result = python_func(py_states);
        
        // 結果を変換
        std::vector<UTTT::InferenceResult> cpp_results;
        for (auto item : result) {
            auto tuple = item.cast<py::tuple>();
            
            // policy (numpy array or list -> std::vector<float>)
            auto policy_obj = tuple[0];
            std::vector<float> policy;
            if (py::isinstance<py::array>(policy_obj)) {
                auto policy_arr = policy_obj.cast<py::array_t<float>>();
                auto buf = policy_arr.request();
                float* ptr = static_cast<float*>(buf.ptr);
                policy.assign(ptr, ptr + buf.size);
            } else {
                policy = policy_obj.cast<std::vector<float>>();
            }
            
            // value (float)
            float value = tuple[1].cast<float>();
            
            cpp_results.push_back({policy, value});
        }
        
        return cpp_results;
    };
}

PYBIND11_MODULE(uttt_cpp, m) {
    m.doc() = "Ultimate Tic-Tac-Toe C++ implementation with Python bindings";
    
    // State クラス
    py::class_<UTTT::State>(m, "State")
        .def(py::init<>())
        .def(py::init<
            const std::array<std::array<int, 9>, 9>&,
            const std::array<std::array<int, 9>, 9>&,
            const std::array<int, 9>&,
            const std::array<int, 9>&,
            int>())
        .def("is_lose", &UTTT::State::is_lose)
        .def("is_draw", &UTTT::State::is_draw)
        .def("is_done", &UTTT::State::is_done)
        .def("is_first_player", &UTTT::State::is_first_player)
        .def("next", &UTTT::State::next)
        .def("legal_actions", &UTTT::State::legal_actions)
        .def("to_string", &UTTT::State::to_string)
        .def("__str__", &UTTT::State::to_string)
        .def("to_input_tensor", &UTTT::State::to_input_tensor)
        .def_property_readonly("pieces", &UTTT::State::get_pieces)
        .def_property_readonly("enemy_pieces", &UTTT::State::get_enemy_pieces)
        .def_property_readonly("main_board_pieces", &UTTT::State::get_main_board_pieces)
        .def_property_readonly("main_board_enemy_pieces", &UTTT::State::get_main_board_enemy_pieces)
        .def_property_readonly("active_board", &UTTT::State::get_active_board);
    
    // InferenceResult 構造体
    py::class_<UTTT::InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("policy", &UTTT::InferenceResult::policy)
        .def_readwrite("value", &UTTT::InferenceResult::value);
    
    // MCTS関数
    m.def("pv_mcts_scores", 
        [](py::object model, const UTTT::State& state, float temperature, 
           int evaluate_count, int batch_size) {
            return UTTT::pv_mcts_scores(
                wrap_python_inference(model), 
                state, 
                temperature, 
                evaluate_count, 
                batch_size
            );
        },
        py::arg("model"),
        py::arg("state"),
        py::arg("temperature") = 0.0f,
        py::arg("evaluate_count") = 50,
        py::arg("batch_size") = 8,
        "Run MCTS and return score distribution over legal actions"
    );
    
    m.def("boltzman", &UTTT::boltzman,
        py::arg("xs"),
        py::arg("temperature"),
        "Apply Boltzmann distribution"
    );
}
