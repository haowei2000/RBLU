#!/bin/bash

# 查找所有包含 _reverse_reverse 的文件和文件夹
find /workspace/project/RBLU/src/score -depth -name '*_reverse_en*' | while read -r file; do
    # 构造新的文件名
    new_name=$(echo "$file" | sed 's/_reverse_en/_en_reverse/g')

    # 确保新旧文件名不同，避免 mv 出错
    if [[ "$file" != "$new_name" ]]; then
        mv "$file" "$new_name"
        echo "Renamed: $file → $new_name"
    fi
done
