import os

def remove_commas_from_file(file_path):
    """
    指定されたファイルからカンマを削除します。

    Args:
        file_path (str): カンマを削除する対象のファイルパス。
    """
    try:
        # ファイルを読み込みモードで開く（エンコーディングはUTF-8を想定）
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # カンマを削除
        new_content = content.replace(',', '')
        
        # ファイルを書き込みモードで開き、変更内容を保存
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"カンマを削除しました: {file_path}")
    
    except Exception as e:
        print(f"エラーが発生しました ({file_path}): {e}")

def process_folder(folder_path):
    """
    指定されたフォルダおよびサブフォルダ内のすべてのテキストファイルからカンマを削除します。

    Args:
        folder_path (str): 処理対象のフォルダパス。
    """
    # os.walkを使用してフォルダ内を再帰的に探索
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(root, filename)
                remove_commas_from_file(file_path)

if __name__ == "__main__":
    # 処理対象のフォルダパスを指定
    folder_path = r'kyoikukatei/2021'  # 例: r'C:\Users\username\Documents\text_files'
    
    # フォルダが存在するか確認
    if os.path.isdir(folder_path):
        process_folder(folder_path)
        print("全てのファイルの処理が完了しました。")
    else:
        print(f"指定されたフォルダが存在しません: {folder_path}")
