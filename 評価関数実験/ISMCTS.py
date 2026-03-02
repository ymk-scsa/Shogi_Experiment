import random
import math

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # ゲームの状態
        self.parent = parent  # 親ノード
        self.children = {}  # 子ノード (行動: ノード)
        self.visits = 0  # ノードへの訪問回数
        self.score = 0  # ノードの合計スコア

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def select_child(self, exploration_factor=0.7):
        # get_scoreメソッドを使って最良の子を選択
        return max(self.children.values(), key=lambda child: child.get_score(exploration_factor))

    def get_score(self, exploration_factor):
        if self.visits == 0:
            return float('inf')
        exploitation = self.score / self.visits
        # UCB1アルゴリズムに基づく探索項
        exploration = exploration_factor * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def add_child(self, action, child_state):
        child = Node(child_state, self)
        self.children[action] = child
        return child

class BlackjackState:
    def __init__(self, player_hand, dealer_hand, usable_ace):
        self.player_hand = player_hand
        self.dealer_hand = dealer_hand
        self.usable_ace = usable_ace

    def get_legal_moves(self):
        return ['hit', 'stand']

    def clone(self):
        return BlackjackState(self.player_hand[:], self.dealer_hand[:], self.usable_ace)

    def is_terminal(self):
        p_score = self.get_score(self.player_hand)
        d_score = self.get_score(self.dealer_hand)
        return p_score > 21 or (len(self.player_hand) == 2 and p_score == 21) or \
                d_score > 21 or (len(self.dealer_hand) == 2 and d_score == 21)

    def get_score(self, hand):
        score = sum(hand)
        if score <= 11 and 1 in hand and self.usable_ace:
            score += 10
        return score

    def move(self, action):
        new_state = self.clone()
        if action == 'hit':
            new_state.player_hand.append(random.randint(1, 13))
            if new_state.get_score(new_state.player_hand) > 21:
                return new_state, -1
            else:
                return new_state, 0
        elif action == 'stand':
            while new_state.get_score(new_state.dealer_hand) < 17:
                new_state.dealer_hand.append(random.randint(1, 13))

            p_score = new_state.get_score(new_state.player_hand)
            d_score = new_state.get_score(new_state.dealer_hand)

            if d_score > 21 or p_score > d_score:
                return new_state, 1
            elif p_score == d_score:
                return new_state, 0
            else:
                return new_state, -1
        return new_state, 0

def ismcts_search(root_state, num_iterations):
    root_node = Node(root_state)

    for _ in range(num_iterations):
        node = root_node
        state = root_state.clone()
        reward = 0

        # 選択
        while not state.is_terminal() and node.is_fully_expanded():
            node = node.select_child()
            # 木のノードが終局状態ならループを抜ける
            if node.state.is_terminal():
                break
            # 実際にはnode.stateが次の状態を保持している
            state = node.state

        # 展開
        if not state.is_terminal():
            legal_moves = state.get_legal_moves()
            # まだ展開されていない行動を選択
            unexplored_moves = [m for m in legal_moves if m not in node.children]
            if unexplored_moves:
                action = random.choice(unexplored_moves)
                new_state, reward = state.move(action)
                node = node.add_child(action, new_state)
                state = new_state

        # シミュレーション
        while not state.is_terminal():
            state, reward = state.move(random.choice(state.get_legal_moves()))

        # バックプロパゲーション
        curr_node = node
        while curr_node is not None:
            curr_node.visits += 1
            curr_node.score += reward
            curr_node = curr_node.parent

    # 最も訪問回数の多い行動のインデックス（または行動名）を返す
    # ここでは「行動」そのものを返すように修正
    best_action = max(root_node.children.items(), key=lambda item: item[1].visits)[0]
    return best_action

def main():
    initial_player_hand = [random.randint(1, 13) for _ in range(2)]
    initial_dealer_hand = [random.randint(1, 13) for _ in range(2)]
    root_state = BlackjackState(initial_player_hand, initial_dealer_hand, True)

    print("Player's initial hand:", root_state.player_hand)
    print("Dealer's initial hand:", root_state.dealer_hand)

    while not root_state.is_terminal():
        # ismcts_searchは「行動(hit or stand)」を返すように修正済み
        action = ismcts_search(root_state, 1000)
        print(f"Decision: {action}")
        root_state, reward = root_state.move(action)
        print("Player's hand after action:", root_state.player_hand)
        if action == 'stand':
            break

    player_score = root_state.get_score(root_state.player_hand)
    dealer_score = root_state.get_score(root_state.dealer_hand)

    print("--- Final Result ---")
    print(f"Player Score: {player_score} | Dealer Score: {dealer_score}")

    if player_score > 21:
        print("Player busts! Dealer wins.")
    elif dealer_score > 21:
        print("Dealer busts! Player wins.")
    elif player_score > dealer_score:
        print("Player wins!")
    elif player_score < dealer_score:
        print("Dealer wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()