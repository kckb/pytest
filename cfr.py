from copy import copy
from copy import deepcopy # 浅いコピーと深いコピーがある

from tqdm import tqdm # tqdm: 進捗状況や処理状況をプログレスバーで表示する機能
import yaml # YAML: XMLなどに似たデータのフォーマットで、設定ファイルやデータの保存などで使われ、XMLよりも扱いやすい

import logger # 出力された内容をログ、それがファイルに落ちる場合はログファイル、ログを取ることをロギングという(※同ディレクトリ内別pythonファイルで定義している)
from envs.toy_pokers import Node, KuhnPoker #Nodeクラス、KuhnPokerクラスをインポート⇒このコード上でKuhn pokerのインスタンスが作成できる


def update_pi(node: Node, strategy_profile: dict, average_strategy_profile: dict, pi_mi_list: list, pi_i_list: list, true_pi_mi_list: list):
    node.pi = pi_mi_list[node.player] * pi_i_list[node.player]
    node.true_pi_mi = true_pi_mi_list[node.player]
    node.pi_mi = pi_mi_list[node.player]
    node.pi_i = pi_i_list[node.player]

    if node.terminal:
        return
    for action, child_node in node.children.items():
        next_pi_mi_list = copy(pi_mi_list)
        next_pi_i_list = copy(pi_i_list)
        next_true_pi_mi_list = copy(true_pi_mi_list)
        for player in strategy_profile.keys():
            if player == node.player:
                next_pi_i_list[player] *= strategy_profile[node.player][node.information][action]
            else:
                next_pi_mi_list[player] *= strategy_profile[node.player][node.information][action]
                next_true_pi_mi_list[player] *= average_strategy_profile[node.player][node.information][action]
        update_pi(child_node, strategy_profile, average_strategy_profile, next_pi_mi_list, next_pi_i_list, next_true_pi_mi_list)


def update_node_values(node: Node, strategy_profile: dict): # dict: 辞書=キーと値が一対一で結びついたリスト
    node_eu = 0  # ノードに到達した後得られる期待利得
    node.num_updates += 1
    if node.terminal:
        return node.eu
    node.pi_i_sum += node.pi_i
    for action, child_node in node.children.items():
        p = strategy_profile[node.player][node.information][action]
        node.pi_sigma_sum[action] += node.pi_i * p
        node_eu += p * update_node_values(child_node, strategy_profile)
    node.eu = node_eu
    node.cv = node.pi_mi * node_eu
    for action, child_node in node.children.items():
        cfr = node.pi_mi * child_node.eu - node.cv if node.player == 0 else (-1) * (node.pi_mi * child_node.eu - node.cv)
        node.cfr[action] += cfr
    return node_eu


def get_initial_strategy_profile(node: Node, num_players=None, strategy_profile=None):
    if node.terminal:
        return strategy_profile
    if strategy_profile is None:
        strategy_profile = {player: {} for player in range(-1, num_players)}  # chance nodeのために+1する
    if node.information not in strategy_profile[node.player]:
        strategy_profile[node.player][node.information] = {action: 1 / len(node.children) for action in node.children}
    for child in node.children.values():
        strategy_profile = get_initial_strategy_profile(child, strategy_profile=strategy_profile)
    return strategy_profile


def update_strategy(strategy_profile: dict, average_strategy_profile: dict, information_sets: dict):
    for player, information_policy in strategy_profile.items(): # playerとinformation_policy（情報集合？）の要素数だけ繰り返す items: 辞書オブジェクトdictのメソッド
        if player == -1:
            continue # player=-1 -> チャンスノード（ディーラー）なので、次のループへ移動
        for information, strategy in information_policy.items():
            cfr = {}
            for action in strategy.keys():
                cfr_in_info = 0
                average_strategy_denominator = 0
                average_strategy_numerator = 0
                for same_info_node in information_sets[player][information]:
                    cfr_in_info += same_info_node.cfr[action]
                    average_strategy_denominator += same_info_node.pi_i_sum
                    average_strategy_numerator += same_info_node.pi_sigma_sum[action]
                cfr[action] = max(cfr_in_info, 0)
                average_strategy_profile[player][information][action] = average_strategy_numerator / average_strategy_denominator
            cfr_sum = sum([cfr_values for cfr_values in cfr.values()])
            for action in strategy.keys():
                if cfr_sum > 0:
                    strategy[action] = cfr[action] / cfr_sum
                else:
                    strategy[action] = 1 / len(strategy)
    return


def compute_exploitability(node, information_sets, average_strategy_profile, opponent_player): #compute_exploitabilityの中でcompute_exploitabilityを呼び出している??
    """
    exploitability when opponent player chooses best response for current strategy
    """
    if node.terminal:
        return node.eu #eu=expected utility 期待利得を返す
    if node.player == opponent_player:  # choose exploitable strategy
        best_response_value = None
        best_weighted_expected_utility = -float("inf") if opponent_player == 0 else float("inf")  # current best value
        for action, _ in node.children.items():
            current_weighted_expected_utility = 0  # current value of sum of reach_p * expected_utility
            current_node_expected_utility = 0
            for same_info_node in information_sets[node.player][node.information]:
                expected_utility = compute_exploitability(same_info_node.children[action], information_sets, average_strategy_profile, opponent_player)
                if node == same_info_node:
                    current_node_expected_utility = expected_utility
                current_weighted_expected_utility += same_info_node.true_pi_mi * expected_utility
            if opponent_player == 0:
                if best_weighted_expected_utility < current_weighted_expected_utility:
                    best_weighted_expected_utility = current_weighted_expected_utility
                    best_response_value = current_node_expected_utility
            else:
                if best_weighted_expected_utility > current_weighted_expected_utility:
                    best_weighted_expected_utility = current_weighted_expected_utility
                    best_response_value = current_node_expected_utility
        return best_response_value

    else:  # use current strategy
        expected_utility = 0
        for action, child_node in node.children.items():
            p = average_strategy_profile[node.player][node.information][action]
            expected_utility += p * compute_exploitability(child_node, information_sets, average_strategy_profile, opponent_player)
    return expected_utility


def get_exploitability(game, average_strategy_profile):
    exploitability = 0
    for player in range(game.num_players):
        if player == 0:
            exploitability += compute_exploitability(game.root, game.information_sets, average_strategy_profile, player)
        else:
            exploitability += (-1) * compute_exploitability(game.root, game.information_sets, average_strategy_profile, player)
    return exploitability


def check_exploitability(): #デフォルトでは使っていない
    """
    KuhnPoker
    """
    game = KuhnPoker()
    nash_equilibrium = game.get_nash_equilibrium(game.root)
    update_pi(game.root, nash_equilibrium, nash_equilibrium, [1.0 for _ in range(game.num_players + 1)],
              [1.0 for _ in range(game.num_players + 1)], [1.0 for _ in range(game.num_players + 1)])
    exploitability = get_exploitability(game, nash_equilibrium)
    print(exploitability)


def train(num_iter, log_schedule):
    game = KuhnPoker() # インスタンス生成時にゲーム木が作成されます。
    strategy_profile = get_initial_strategy_profile(game.root, game.num_players) # 最初の戦略プロファイル
    average_strategy_profile = deepcopy(strategy_profile) # 平均戦略プロファイルの初期値=最初の戦略プロファイル
    for t in tqdm(range(num_iter)): # tqdm: 進捗状況や処理状況をプログレスバーで表示する機能
        update_pi(game.root, strategy_profile, average_strategy_profile, [1.0 for _ in range(game.num_players + 1)], [1.0 for _ in range(game.num_players + 1)], [1.0 for _ in range(game.num_players + 1)]) # 到達確率の更新
        update_node_values(game.root, strategy_profile) # 各ノードの値の更新
        exploitability = get_exploitability(game, average_strategy_profile) 
        update_strategy(strategy_profile, average_strategy_profile, game.information_sets) # 戦略の更新
        if t % log_schedule(t) == 0:
            logger.logkv("t", t)
            logger.logkv("exploitability", exploitability)
            logger.dumpkvs()
    return average_strategy_profile


def add_dict_to_dict(d: dict, key):
    if key not in d:
        d[key] = {}


def export_strategy_profile_to_yaml(strategy_profile_result):
    result = {}
    for player, sigma in strategy_profile_result.items(): #辞書オブジェクトdictの要素をfor文でループ処理する場合、keys(),values(),items()を使う
        if player == -1:
            continue
        for info, p_dist in sigma.items():
            add_dict_to_dict(result, info[0][0])
            if len(info[1]) == 0:
                history = "-"
            else:
                history = "-".join(info[1])
            for action, p in p_dist.items():
                add_dict_to_dict(result[info[0][0]], history)
                result[info[0][0]][history][action] = p
    with open("../sample_result.yaml", "w") as f:
        yaml.dump(result, f)


def main():
    logger.configure("./logs") #configureはloggerで定義されている
    num_updates = int(5e4)
    average_strategy_profile = train(num_updates, lambda x: (10 ** (len(str(x)) - 1))) #lamda: ラムダ式、無名関数 （def文を使わないで関数を定義）
    export_strategy_profile_to_yaml(average_strategy_profile) # ハンド→履歴(初手の場合は'-')→行動: 選択確率の順で記述


if __name__ == "__main__":
    main()
