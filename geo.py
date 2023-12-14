import json
import re


def _load_file_as_list() -> list:
    """
    Reads the contents of the 'gra.txt' file, filters out empty lines and lines starting with '%',
    and returns the remaining non-empty and non-comment lines as a list.

    Returns:
    - list: A list containing the non-empty and non-comment lines from the 'gra.txt' file.

    Note:
    - The function assumes that the 'gra.txt' file exists in the current working directory.
    - Empty lines (containing only whitespace characters) and lines starting with '%' are excluded.
    """

    with open("gra.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [
        line for line in lines if (line.strip() != "") and (not line.startswith("%"))
    ]

    return lines


def _create_tmp_dict() -> dict:
    """
    Reads the contents of the 'gra.txt' file, extracts headwords and associated texts,
    and organizes them into a dictionary.

    Returns:
    - dict: A dictionary where each headword is a key, and the associated texts are stored as a list.

    Note:
    - The function relies on the '_load_file_as_list' function to obtain the necessary lines from 'gra.txt'.
    - It uses a regular expression to extract headwords marked by '<k1>' and '<k2>'.
    - The extracted headwords are used as keys, and the associated texts are stored as lists in the dictionary.
    """

    lines = _load_file_as_list()

    headword_pattern = re.compile(r"<k1>(.*?)<k2>")

    headword_list = list(map(lambda x: bool(headword_pattern.search(x)), lines))

    dictionary_dict_tmp = {}
    for i, flag in enumerate(headword_list):
        if flag:
            headword = headword_pattern.search(lines[i]).group(1)

            if not headword in dictionary_dict_tmp:
                dictionary_dict_tmp[headword] = []

        else:
            text = lines[i]
            dictionary_dict_tmp[headword].append(text)

    return dictionary_dict_tmp


def create_dict() -> dict:
    """
    Reads the contents of the 'gra.txt' file, extracts structured information about headwords,
    descriptions, endings, and their meanings, and organizes them into a dictionary.

    Returns:
    - dict: A structured dictionary containing information about headwords, descriptions, endings,
            and their associated meanings and occurrences.

    Note:
    - The function relies on the '_create_tmp_dict' function to obtain a preliminary dictionary
      with headwords and associated texts.
    - It structures the 'description' field, separating accentuated headwords and meanings with numbers.
    - It handles multiple meanings and their occurrences for each ending, organizing them into a dictionary.
    """

    dictionary_dict_tmp = _create_tmp_dict()

    # ただのテキストであるdescriptionを構造化する
    dictionary_dict = {}
    for key, value in dictionary_dict_tmp.items():
        # 1行目に ¦ が含まれていると意味記述がある
        if "¦" in value[0]:
            dictionary_dict[key] = {}

            # ¦よりまえがアクセント付き見出し語、それより後ろが意味記述
            hw, description = value[0].split("¦", 1)

            # 意味を番号で分ける
            meanings = re.split(r"\d+〉", description.strip())
            # 番号をキーとしたディクショナリにする
            meaning_dict = {i: item.strip() for i, item in enumerate(meanings)}

            # head にはアクセントつき見出し語、meanings には 数字と意味がキー・バリューのディクショナリ
            dictionary_dict[key]["text"] = {"head": hw, "meanings": meaning_dict}
            # value[0] = {"head": hw, "meanings": meaning_dict}
        else:
            continue

        # 2行目以降は <div> タグを持ち、語尾と意味番号、出現箇所がのっている
        dictionary_dict[key]["endings"] = []
        for text in value[1:]:
            # for meaning_number, text in enumerate(value[1:]):
            if "<div" in text:
                # divタグを取り除く
                text = re.sub(r"<div.*?>", "", text)

                # 数字 + 〉で分ける
                meaning_occurrence_list = re.split(r"(\d+〉)", text)

                # 0番目の要素は語尾、それより後ろが意味番号と出現箇所
                ending, *meaning_occurrence_list = meaning_occurrence_list

                # 語尾の<ab>タグを消す。
                ending = re.sub(r"</*ab>", "", ending)
                # 余計な空白を消す
                ending = re.sub(r"\s", "", ending)

                # [意味, 出現箇所, 意味, 出現箇所, ...] となっているリストをディクショナリにする
                meaning_occurrence_dict = {
                    meaning_occurrence_list[i]: meaning_occurrence_list[i + 1]
                    for i in range(0, len(meaning_occurrence_list) - 1, 2)
                }

                # 意味番号は数字だけに、出現箇所は {} に囲まれた数字の配列にする
                meaning_occurrence_dict = {
                    n[:-1]: re.findall(r"{(.*?)}", occurrence)
                    for n, occurrence in meaning_occurrence_dict.items()
                }

                dictionary_dict[key]["endings"].append(
                    {
                        "ending": ending.strip(),
                        "occurrences": meaning_occurrence_dict,
                    }
                )

    return dictionary_dict


def save_as_json(d: dict, filename: str) -> None:
    """
    Saves a dictionary as a JSON file.

    Parameters:
    - d (dict): The dictionary to be saved.
    - filename (str): The name of the JSON file to be created.

    Returns:
    - None

    Note:
    - The function writes the dictionary 'd' to a JSON file with specified indentation.
    """

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def create_json():
    dictionary_dict = create_dict()

    json_path = "gra.json"
    save_as_json(dictionary_dict, json_path)


def load_json(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def find_proper_noun() -> dict:
    dictionary_dict = load_json("gra.json")
    result_dict = {}
    errors = []

    for headword, text_endings in dictionary_dict.items():
        text = text_endings["text"]
        endings = text_endings["endings"]
        eigenname_list = []

        for meaning in text["meanings"].values():
            if "eigenname" in meaning.lower():
                eigenname_list.append(True)
            else:
                eigenname_list.append(False)

        for ending_dict in endings:
            meaningnumber_occurrences_dict = ending_dict["occurrences"]

            for meaning_number, occurrences in meaningnumber_occurrences_dict.items():
                try:
                    if eigenname_list[int(meaning_number)]:
                        print(headword)
                        try:
                            result_dict[headword].extend(occurrences)
                        except KeyError:
                            result_dict[headword] = []
                            result_dict[headword].extend(occurrences)

                except IndexError:
                    errors.append(
                        [headword, "".join(text["meanings"].values()), meaning_number]
                    )

    _save_errors(errors)
    return result_dict


def _save_errors(errors: list):
    filename = "error.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)


import pandas as pd
import rdflib
from rdflib import Graph, Literal, Namespace


def _serial_to_poet(n: int) -> set:
    df = pd.read_csv("rv_info.csv")

    poet_column = df[df["serNum"] == n]["poet"]

    poet_list = poet_column.to_list()

    return set(poet_list)


# result.json の出現箇所から詩人の名前を取ってきて、 {{poet}} refers {{properNoun}} という形式にしたい
def create_rdf() -> rdflib.graph.Graph:
    with open("result.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # RDFグラフを作成
    g = Graph()

    # Namespaceを設定
    ex = Namespace("ex:")

    # jsonから、詩人: 固有名詞のディクショナリを作る
    data = {}
    for key, values in json_data.items():
        for value in values:
            poet_set = _serial_to_poet(int(value.split(",")[0]))
            poet_name = "/".join(poet_set)
            try:
                data[poet_name].append(key)
            except KeyError:
                data[poet_name] = []
                data[poet_name].append(key)

    # JSONデータをRDFに変換
    for poet, nouns in data.items():
        for noun in nouns:
            g.add((ex[re.sub(r"\s", "_", poet)], ex["refers"], ex[noun]))

    # RDFグラフをTurtle形式で表示
    with open("rdf.txt", "w", encoding="utf-8") as f:
        print(g.serialize(format="turtle"))
        f.write(g.serialize(format="turtle"))

    return g.serialize(format="turtle")


def visualize():
    from rdflib import Graph, Literal, Namespace

    # 仮のRDFグラフを作成
    g = Graph()

    # 仮のTurtle形式データ (A refers B)
    turtle_data = create_rdf()

    # RDFグラフにTurtle形式データを追加
    g.parse(data=turtle_data, format="turtle")

    import matplotlib.pyplot as plt
    # A同士の関係を可視化するためのネットワーク図を作成
    import networkx as nx

    # 空の有向グラフを作成
    G = nx.DiGraph()

    # ノードを追加
    for subject, _, _ in g:
        G.add_node(subject)

    # エッジを追加し、同じBに言及する回数を重みとして設定
    edge_weights = {}
    ex = Namespace("ex:")

    for subject, predicate, obj in g:
        if predicate == ex.refers:
            subject = str(subject)
            obj = str(obj)
            edge = (subject, obj)
            if edge in edge_weights:
                edge_weights[edge] += 1
            else:
                edge_weights[edge] = 1

    # ネットワーク図を描画
    pos = nx.spring_layout(G)

    # エッジの太さを重みに応じて設定
    edge_widths = [edge_weights[edge] for edge in G.edges]

    # 図を描画
    nx.draw(
        G,
        pos,
        with_labels=True,
        font_weight="bold",
        node_size=700,
        node_color="skyblue",
        font_size=8,
        arrowsize=10,
        width=edge_widths,
        edge_color="gray",
    )

    # 図を表示
    plt.savefig("network.png")


def main():
    # create_json()
    # d = find_proper_noun()
    # save_as_json(d, "result.json")
    # create_rdf()
    visualize()


if __name__ == "__main__":
    main()
