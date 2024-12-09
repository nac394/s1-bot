def is_all_strings_in_file(filename: str) -> bool:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # すべての行が文字列であるかを確認
            for line in lines:
                # 各行が文字列であるか確認
                if not isinstance(line, str):
                    return False
        return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    # チェックしたいファイル名を指定
    filename = "kanren/kanren-sample.txt"
    
    if is_all_strings_in_file(filename):
        print(f"{filename} 内のすべての行は文字列です。")
    else:
        print(f"{filename} 内に文字列ではない行があります。")
