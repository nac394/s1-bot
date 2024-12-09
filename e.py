import os
from pathlib import Path

def extract_first_lines_per_subfolder(parent_folder, output_suffix='_title.txt'):
    """
    親フォルダ内の各サブフォルダごとに、CSVファイルの1行目を抽出し、
    各サブフォルダに対応するテキストファイルにまとめる関数。

    :param parent_folder: 親フォルダのパス
    :param output_suffix: 出力テキストファイルのサフィックス（デフォルトは '_first_lines.txt'）
    """
    parent_path = Path(parent_folder)

    # 親フォルダ内のすべてのサブフォルダを取得
    for subfolder in parent_path.iterdir():
        if subfolder.is_dir():
            output_file = subfolder / f"{subfolder.name}{output_suffix}"
            with output_file.open('w', encoding='utf-8') as outfile:
                # サブフォルダ内のすべてのCSVファイルを検索
                for csv_file in subfolder.glob('*.csv'):
                    try:
                        with csv_file.open('r', encoding='utf-8') as infile:
                            first_line = infile.readline().strip()
                            first_line_no_commas = first_line.replace(',', '')
                            # ファイル名と1行目を出力ファイルに書き込む
                            outfile.write(f"{first_line}\n")
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
            print(f"{subfolder.name} の1行目を {output_file} に書き込みました。")

if __name__ == "__main__":
    # 親フォルダのパスを指定
    parent_folder = 'kyoikukatei/2021'
    
    extract_first_lines_per_subfolder(parent_folder)
    print("すべてのサブフォルダのCSVファイルの1行目を抽出しました。")
