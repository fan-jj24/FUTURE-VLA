#!/bin/bash

# 读取total_cfg.yaml中的配置
NUMS_VIEW=$(grep "nums_view:" total_cfg.yaml | awk '{print $2}')
COMPRESS_FRAME=$(grep "compress_frame:" total_cfg.yaml | cut -d: -f2 | tr -d ' ' | sed 's/\[//' | sed 's/\]//' | sed 's/,/, /g')

# 检查是否成功读取配置
if [ -z "$NUMS_VIEW" ] || [ -z "$COMPRESS_FRAME" ]; then
    echo "Error: Failed to read configuration from total_cfg.yaml"
    exit 1
fi

echo "Read configuration:"
echo "  nums_view: $NUMS_VIEW"
echo "  COMPRESS_FRAME: [$COMPRESS_FRAME]"

# 检查第505行格式是否符合预期（20个空格 + media_inputs = processor.deeper_processor）
LINE_505=$(sed -n '505p' replace_code/swift/qwen.py)
EXPECTED_PREFIX="                    media_inputs = processor.deeper_processor(images=mm_data, grid_config = "

echo -e "\033[0;34mmodify qwen.py:\033[0m"

if [[ ! "$LINE_505" =~ ^"$EXPECTED_PREFIX" ]]; then
    echo -e "\033[0;31m  Line 505 format check failed!\033[0m"
    echo "  Expected prefix: '${EXPECTED_PREFIX}'"
    echo "  Actual line: '$LINE_505'"
else
    echo -e "\033[0;32m  Line 505 format check passed.\033[0m"
    
    # 更新replace_code/swift/qwen.py的第505行（注意20个空格缩进）
    sed -i "505c\\                    media_inputs = processor.deeper_processor(images=mm_data, grid_config = [$NUMS_VIEW, [$COMPRESS_FRAME]])" replace_code/swift/qwen.py
    
    echo -e "\033[0;32m  Updated replace_code/swift/qwen.py line 505 successfully!\033[0m"
fi
