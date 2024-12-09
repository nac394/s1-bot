import os

def remove_character_from_filenames(folder_path, character_to_remove):
    try:
        # 指定フォルダ内のすべてのファイルとディレクトリを取得
        files = os.listdir(folder_path)
        
        for filename in files:
            # フルパスを取得
            full_path = os.path.join(folder_path, filename)
            
            # ファイルかどうか確認
            if os.path.isfile(full_path):
                # 特定の文字を削除した新しいファイル名を作成
                new_filename = filename.replace(character_to_remove, "")
                
                # 新しいフルパスを生成
                new_full_path = os.path.join(folder_path, new_filename)
                
                # ファイル名を変更
                os.rename(full_path, new_full_path)
                print(f"Renamed: {filename} -> {new_filename}")
    except Exception as e:
        print(f"Error: {e}")

# 使用例
folder_path = "kyoikukateihyo/2022"  # 変更するフォルダパスを指定
character_to_remove = "2022_"  # 削除したい文字を指定
remove_character_from_filenames(folder_path, character_to_remove)
