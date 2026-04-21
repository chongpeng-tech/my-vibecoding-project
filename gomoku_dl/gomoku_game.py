import math
import random
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Optional, Sequence, Tuple

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1  # 玩家
WHITE = 2  # AI
SEARCH_DEPTH = 4

# 棋型评分（按需求）
PATTERN_SCORES = {
    "FIVE": 1_000_000,
    "OPEN_FOUR": 100_000,
    "CLOSED_FOUR": 10_000,
    "OPEN_THREE": 5_000,
    "SLEEP_THREE": 500,
    "OPEN_TWO": 100,
}

# Threat Space Search 的优先等级
THREAT_PRIORITY = {
    "OPEN_FOUR": 5,
    "CLOSED_FOUR": 4,
    "OPEN_THREE": 3,
    "SLEEP_THREE": 2,
}

# 方向向量
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


class DeepPolicyNet:
    """简化版“深度策略网络”（纯 Python 推理）。

    这里不依赖外部深度学习框架，使用多层感知机结构对候选点做打分。
    作用：为 Minimax 的候选走子提供先验排序，让搜索更像“深度学习 + 博弈树”混合策略。
    """

    def __init__(self) -> None:
        random.seed(42)
        # 输入特征: [my_attack, my_defense, opp_attack, opp_defense, center_bias, neighbor_density]
        self.w1 = [
            [0.9, 0.2, -0.5, 0.3, 0.4, 0.7],
            [0.2, 0.9, 0.3, -0.5, 0.4, 0.7],
            [0.7, 0.7, 0.6, 0.6, 0.4, 0.2],
            [1.2, -0.1, -0.4, 0.6, 0.1, 0.8],
            [0.6, 0.5, 0.5, 0.6, 0.4, 0.5],
            [0.4, 0.4, 0.4, 0.4, 0.8, 0.2],
            [1.0, 0.3, -0.3, 0.7, 0.2, 0.6],
            [0.3, 1.0, 0.7, -0.2, 0.2, 0.6],
        ]
        self.b1 = [0.2, 0.2, 0.1, 0.3, 0.1, 0.0, 0.2, 0.2]
        self.w2 = [0.7, 0.7, 0.5, 0.9, 0.6, 0.3, 0.8, 0.8]
        self.b2 = 0.1

    @staticmethod
    def relu(v: float) -> float:
        return max(0.0, v)

    @staticmethod
    def sigmoid(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-v))

    def predict(self, features: Sequence[float]) -> float:
        hidden = []
        for i, row in enumerate(self.w1):
            s = self.b1[i]
            for w, x in zip(row, features):
                s += w * x
            hidden.append(self.relu(s))
        out = self.b2 + sum(w * h for w, h in zip(self.w2, hidden))
        return self.sigmoid(out)


class GomokuAI:
    def __init__(self) -> None:
        self.policy = DeepPolicyNet()
        # 简化开局库（白棋）
        self.opening_book = {
            1: [(7, 7)],  # 若 AI 先手（当前项目中玩家先手，该项做完整性保留）
            2: [(7, 8), (8, 7), (6, 7), (7, 6)],  # 玩家首手后
            4: [(8, 8), (6, 6), (8, 6), (6, 8)],
        }

    def get_best_move(self, board: List[List[int]], move_history: List[Tuple[int, int, int]]) -> Tuple[int, int]:
        opening_move = self._try_opening_book(board, move_history)
        if opening_move:
            return opening_move

        candidates = self.generate_candidates(board, WHITE)
        if not candidates:
            return BOARD_SIZE // 2, BOARD_SIZE // 2

        best_score = -float("inf")
        best_move = candidates[0]

        alpha = -float("inf")
        beta = float("inf")

        for x, y in candidates:
            board[x][y] = WHITE
            score = self.minimax(board, SEARCH_DEPTH - 1, alpha, beta, False)
            board[x][y] = EMPTY

            if score > best_score:
                best_score = score
                best_move = (x, y)
            alpha = max(alpha, best_score)

        return best_move

    def minimax(self, board: List[List[int]], depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        winner = self.check_winner(board)
        if winner == WHITE:
            return 9_999_999 + depth
        if winner == BLACK:
            return -9_999_999 - depth
        if depth == 0:
            return self.evaluate_board(board)

        current = WHITE if maximizing else BLACK
        candidates = self.generate_candidates(board, current)
        if not candidates:
            return self.evaluate_board(board)

        if maximizing:
            value = -float("inf")
            for x, y in candidates:
                board[x][y] = WHITE
                value = max(value, self.minimax(board, depth - 1, alpha, beta, False))
                board[x][y] = EMPTY
                alpha = max(alpha, value)
                if alpha >= beta:  # Alpha-Beta 剪枝
                    break
            return value
        else:
            value = float("inf")
            for x, y in candidates:
                board[x][y] = BLACK
                value = min(value, self.minimax(board, depth - 1, alpha, beta, True))
                board[x][y] = EMPTY
                beta = min(beta, value)
                if alpha >= beta:  # Alpha-Beta 剪枝
                    break
            return value

    def generate_candidates(self, board: List[List[int]], current_player: int) -> List[Tuple[int, int]]:
        # 1) Threat Space Search：优先应对高威胁
        threat_moves = self.find_threat_moves(board, current_player)
        if threat_moves:
            return threat_moves

        # 2) 常规候选：取已有棋子附近空位
        candidates = set()
        has_piece = False
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] != EMPTY:
                    has_piece = True
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if in_bounds(nx, ny) and board[nx][ny] == EMPTY:
                                candidates.add((nx, ny))

        if not has_piece:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

        scored: List[Tuple[float, Tuple[int, int]]] = []
        for x, y in candidates:
            # 落子进攻收益
            board[x][y] = current_player
            my_score = self.evaluate_point(board, x, y, current_player)
            board[x][y] = EMPTY

            opp = BLACK if current_player == WHITE else WHITE
            # 落子防守收益（阻断对方）
            board[x][y] = opp
            opp_score = self.evaluate_point(board, x, y, opp)
            board[x][y] = EMPTY

            features = self._build_features(board, x, y, current_player, my_score, opp_score)
            prior = self.policy.predict(features)
            # 综合评分：棋型 + 深度先验
            total = my_score * 1.0 + opp_score * 0.9 + prior * 600
            scored.append((total, (x, y)))

        scored.sort(reverse=True, key=lambda t: t[0])
        return [m for _, m in scored[:14]]

    def find_threat_moves(self, board: List[List[int]], current_player: int) -> List[Tuple[int, int]]:
        opp = BLACK if current_player == WHITE else WHITE
        bucket: List[Tuple[int, Tuple[int, int]]] = []

        candidates = set()
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] != EMPTY:
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if in_bounds(nx, ny) and board[nx][ny] == EMPTY:
                                candidates.add((nx, ny))

        if not candidates:
            return []

        for x, y in candidates:
            # 当前方制造威胁
            board[x][y] = current_player
            ptype = self.best_pattern_type(board, x, y, current_player)
            board[x][y] = EMPTY
            if ptype in THREAT_PRIORITY:
                bucket.append((THREAT_PRIORITY[ptype] + 10, (x, y)))

            # 阻断对手威胁
            board[x][y] = opp
            otype = self.best_pattern_type(board, x, y, opp)
            board[x][y] = EMPTY
            if otype in THREAT_PRIORITY:
                bucket.append((THREAT_PRIORITY[otype] + 7, (x, y)))

        if not bucket:
            return []

        bucket.sort(reverse=True, key=lambda t: t[0])
        uniq = []
        seen = set()
        for _, m in bucket:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq[:10]

    def evaluate_board(self, board: List[List[int]]) -> float:
        white_score = 0
        black_score = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == WHITE:
                    white_score += self.evaluate_point(board, x, y, WHITE)
                elif board[x][y] == BLACK:
                    black_score += self.evaluate_point(board, x, y, BLACK)
        return white_score - black_score * 1.05

    def evaluate_point(self, board: List[List[int]], x: int, y: int, player: int) -> int:
        score = 0
        for dx, dy in DIRECTIONS:
            score += self.evaluate_direction(board, x, y, dx, dy, player)
        return score

    def evaluate_direction(self, board: List[List[int]], x: int, y: int, dx: int, dy: int, player: int) -> int:
        count = 1
        open_ends = 0

        # 正方向
        i = 1
        while in_bounds(x + i * dx, y + i * dy) and board[x + i * dx][y + i * dy] == player:
            count += 1
            i += 1
        if in_bounds(x + i * dx, y + i * dy) and board[x + i * dx][y + i * dy] == EMPTY:
            open_ends += 1

        # 反方向
        i = 1
        while in_bounds(x - i * dx, y - i * dy) and board[x - i * dx][y - i * dy] == player:
            count += 1
            i += 1
        if in_bounds(x - i * dx, y - i * dy) and board[x - i * dx][y - i * dy] == EMPTY:
            open_ends += 1

        if count >= 5:
            return PATTERN_SCORES["FIVE"]
        if count == 4:
            if open_ends == 2:
                return PATTERN_SCORES["OPEN_FOUR"]
            if open_ends == 1:
                return PATTERN_SCORES["CLOSED_FOUR"]
        if count == 3:
            if open_ends == 2:
                return PATTERN_SCORES["OPEN_THREE"]
            if open_ends == 1:
                return PATTERN_SCORES["SLEEP_THREE"]
        if count == 2 and open_ends == 2:
            return PATTERN_SCORES["OPEN_TWO"]
        return 0

    def best_pattern_type(self, board: List[List[int]], x: int, y: int, player: int) -> str:
        best = ""
        best_score = 0
        for dx, dy in DIRECTIONS:
            count = 1
            open_ends = 0

            i = 1
            while in_bounds(x + i * dx, y + i * dy) and board[x + i * dx][y + i * dy] == player:
                count += 1
                i += 1
            if in_bounds(x + i * dx, y + i * dy) and board[x + i * dx][y + i * dy] == EMPTY:
                open_ends += 1

            i = 1
            while in_bounds(x - i * dx, y - i * dy) and board[x - i * dx][y - i * dy] == player:
                count += 1
                i += 1
            if in_bounds(x - i * dx, y - i * dy) and board[x - i * dx][y - i * dy] == EMPTY:
                open_ends += 1

            if count >= 5:
                return "OPEN_FOUR"
            if count == 4 and open_ends == 2:
                t = "OPEN_FOUR"
            elif count == 4 and open_ends == 1:
                t = "CLOSED_FOUR"
            elif count == 3 and open_ends == 2:
                t = "OPEN_THREE"
            elif count == 3 and open_ends == 1:
                t = "SLEEP_THREE"
            else:
                t = ""

            if t:
                sc = THREAT_PRIORITY[t]
                if sc > best_score:
                    best_score = sc
                    best = t
        return best

    def check_winner(self, board: List[List[int]]) -> int:
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                player = board[x][y]
                if player == EMPTY:
                    continue
                for dx, dy in DIRECTIONS:
                    cnt = 1
                    nx, ny = x + dx, y + dy
                    while in_bounds(nx, ny) and board[nx][ny] == player:
                        cnt += 1
                        if cnt >= 5:
                            return player
                        nx += dx
                        ny += dy
        return EMPTY

    def _build_features(
        self,
        board: List[List[int]],
        x: int,
        y: int,
        current_player: int,
        my_score: float,
        opp_score: float,
    ) -> List[float]:
        opp = BLACK if current_player == WHITE else WHITE
        center = (BOARD_SIZE - 1) / 2
        center_bias = 1.0 - (abs(x - center) + abs(y - center)) / (BOARD_SIZE)

        density = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny) and board[nx][ny] != EMPTY:
                    density += 1

        # 防守时同样考虑该点如果被对方占据的攻击力
        board[x][y] = opp
        opp_attack = self.evaluate_point(board, x, y, opp)
        board[x][y] = EMPTY

        my_attack = my_score / 100000
        my_defense = opp_score / 100000
        opp_attack_norm = opp_attack / 100000
        opp_defense = my_score / 120000

        return [my_attack, my_defense, opp_attack_norm, opp_defense, center_bias, density / 25]

    def _try_opening_book(
        self,
        board: List[List[int]],
        move_history: List[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int]]:
        steps = len(move_history) + 1
        if steps not in self.opening_book:
            return None

        for x, y in self.opening_book[steps]:
            if in_bounds(x, y) and board[x][y] == EMPTY:
                return (x, y)
        return None


class GomokuGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("五子棋人机对战（Minimax + Alpha-Beta + Threat Search）")

        self.cell = 38
        self.margin = 36
        self.board_px = self.margin * 2 + self.cell * (BOARD_SIZE - 1)

        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.move_history: List[Tuple[int, int, int]] = []
        self.current_player = BLACK
        self.game_over = False
        self.ai = GomokuAI()
        self.last_move: Optional[Tuple[int, int]] = None
        self.turn_count = 1
        self.ai_thinking = False

        self.status_var = tk.StringVar(value="轮到你（黑棋）")

        self._build_ui()
        self.draw_board()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#d7c6a5")
        top.pack(fill="x")

        title = tk.Label(
            top,
            text="五子棋 · 人机对战",
            font=("Microsoft YaHei", 16, "bold"),
            bg="#d7c6a5",
            fg="#3a2d18",
            pady=8,
        )
        title.pack()

        info_frame = tk.Frame(self.root, bg="#f4ead3")
        info_frame.pack(fill="x")

        self.turn_var = tk.StringVar(value="第 1 手")
        tk.Label(info_frame, textvariable=self.turn_var, font=("Microsoft YaHei", 11), bg="#f4ead3").pack(side="left", padx=12, pady=8)
        tk.Label(info_frame, textvariable=self.status_var, font=("Microsoft YaHei", 11), bg="#f4ead3").pack(side="left", padx=12)

        btn_frame = tk.Frame(info_frame, bg="#f4ead3")
        btn_frame.pack(side="right", padx=10)

        tk.Button(btn_frame, text="悔棋", command=self.undo_move, width=8, bg="#eee4cc").pack(side="left", padx=5)
        tk.Button(btn_frame, text="重新开始", command=self.restart_game, width=10, bg="#eee4cc").pack(side="left", padx=5)

        self.canvas = tk.Canvas(
            self.root,
            width=self.board_px,
            height=self.board_px,
            bg="#c9a76a",
            highlightthickness=0,
        )
        self.canvas.pack(padx=16, pady=14)
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self) -> None:
        self.canvas.delete("all")

        # 木纹背景（简化：渐变条纹）
        for i in range(self.board_px):
            tone = 165 + int(20 * math.sin(i / 14.0))
            tone = max(130, min(210, tone))
            color = f"#{tone:02x}{int(tone*0.8):02x}{int(tone*0.45):02x}"
            self.canvas.create_line(0, i, self.board_px, i, fill=color)

        # 网格线
        for i in range(BOARD_SIZE):
            x = self.margin + i * self.cell
            self.canvas.create_line(self.margin, x, self.board_px - self.margin, x, fill="#3f2f1e", width=1)
            self.canvas.create_line(x, self.margin, x, self.board_px - self.margin, fill="#3f2f1e", width=1)

        # 天元和星位
        stars = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        for sx, sy in stars:
            px, py = self.to_pixel(sx, sy)
            self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#2f2216", outline="")

        # 棋子
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x][y] != EMPTY:
                    self.draw_stone(x, y, self.board[x][y], animate=False)

        # 最后一手高亮
        if self.last_move:
            lx, ly = self.last_move
            px, py = self.to_pixel(lx, ly)
            self.canvas.create_rectangle(px - 9, py - 9, px + 9, py + 9, outline="#ff5252", width=2)

    def draw_stone(self, x: int, y: int, player: int, animate: bool = True) -> None:
        px, py = self.to_pixel(x, y)

        def draw_at_radius(r: float) -> None:
            tag = f"stone_{x}_{y}"
            self.canvas.delete(tag)
            if player == BLACK:
                self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="#1f1f1f", outline="#000000", width=1.5, tags=tag)
                self.canvas.create_oval(px - r * 0.6, py - r * 0.65, px - r * 0.05, py - r * 0.1, fill="#666666", outline="", tags=tag)
            else:
                self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="#f9f9f9", outline="#cccccc", width=1.5, tags=tag)
                self.canvas.create_oval(px - r * 0.55, py - r * 0.6, px - r * 0.1, py - r * 0.15, fill="#ffffff", outline="", tags=tag)

        if not animate:
            draw_at_radius(14)
            return

        steps = [5, 8, 11, 14]

        def animate_step(idx: int = 0) -> None:
            if idx >= len(steps):
                return
            draw_at_radius(steps[idx])
            self.root.after(24, lambda: animate_step(idx + 1))

        animate_step(0)

    def to_board(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        x = round((px - self.margin) / self.cell)
        y = round((py - self.margin) / self.cell)
        if not in_bounds(x, y):
            return None
        cx, cy = self.to_pixel(x, y)
        if abs(cx - px) <= self.cell * 0.45 and abs(cy - py) <= self.cell * 0.45:
            return x, y
        return None

    def to_pixel(self, x: int, y: int) -> Tuple[int, int]:
        return self.margin + x * self.cell, self.margin + y * self.cell

    def on_click(self, event: tk.Event) -> None:
        if self.game_over or self.current_player != BLACK or self.ai_thinking:
            return
        pos = self.to_board(event.x, event.y)
        if not pos:
            return
        x, y = pos
        if self.board[x][y] != EMPTY:
            return
        self.place_move(x, y, BLACK)

        if self.check_end(BLACK):
            return

        self.current_player = WHITE
        self.status_var.set("电脑思考中…")
        self.ai_thinking = True
        # 使用 after + 线程，类似 setTimeout 异步处理，避免界面卡顿
        self.root.after(50, self.trigger_ai_move)

    def trigger_ai_move(self) -> None:
        def worker() -> None:
            move = self.ai.get_best_move(self.board, self.move_history)
            self.root.after(0, lambda: self.on_ai_move_ready(move))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def on_ai_move_ready(self, move: Tuple[int, int]) -> None:
        if self.game_over:
            self.ai_thinking = False
            return

        x, y = move
        if not in_bounds(x, y) or self.board[x][y] != EMPTY:
            # 兜底：找任意空位
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == EMPTY:
                        x, y = i, j
                        break
                else:
                    continue
                break

        self.place_move(x, y, WHITE)
        self.ai_thinking = False

        if self.check_end(WHITE):
            return

        self.current_player = BLACK
        self.status_var.set("轮到你（黑棋）")

    def place_move(self, x: int, y: int, player: int) -> None:
        self.board[x][y] = player
        self.move_history.append((x, y, player))
        self.last_move = (x, y)
        self.turn_count += 1
        self.turn_var.set(f"第 {len(self.move_history)} 手")
        self.draw_board()
        self.draw_stone(x, y, player, animate=True)
        self.draw_last_highlight()

    def draw_last_highlight(self) -> None:
        if not self.last_move:
            return
        lx, ly = self.last_move
        px, py = self.to_pixel(lx, ly)
        self.canvas.create_rectangle(px - 9, py - 9, px + 9, py + 9, outline="#ff5252", width=2)

    def check_end(self, player: int) -> bool:
        winner = self.ai.check_winner(self.board)
        if winner != EMPTY:
            self.game_over = True
            if winner == BLACK:
                self.status_var.set("你获胜了！")
                messagebox.showinfo("游戏结束", "恭喜，你赢了！")
            else:
                self.status_var.set("电脑获胜")
                messagebox.showinfo("游戏结束", "电脑赢了，再接再厉！")
            return True

        if all(self.board[x][y] != EMPTY for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)):
            self.game_over = True
            self.status_var.set("平局")
            messagebox.showinfo("游戏结束", "平局！")
            return True

        return False

    def undo_move(self) -> None:
        if self.ai_thinking:
            return
        if not self.move_history or self.game_over and len(self.move_history) == 0:
            return

        # 人机对战悔棋：默认回退两步（玩家 + 电脑）
        steps = 2 if len(self.move_history) >= 2 else 1
        for _ in range(steps):
            x, y, _ = self.move_history.pop()
            self.board[x][y] = EMPTY

        self.last_move = self.move_history[-1][:2] if self.move_history else None
        self.game_over = False
        self.current_player = BLACK
        self.turn_var.set(f"第 {len(self.move_history) + 1} 手")
        self.status_var.set("轮到你（黑棋）")
        self.draw_board()

    def restart_game(self) -> None:
        if self.ai_thinking:
            return
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.move_history.clear()
        self.current_player = BLACK
        self.game_over = False
        self.last_move = None
        self.turn_count = 1
        self.turn_var.set("第 1 手")
        self.status_var.set("轮到你（黑棋）")
        self.draw_board()


def main() -> None:
    root = tk.Tk()
    app = GomokuGUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
