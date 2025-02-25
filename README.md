brew install anaconda OR winget install anaconda
conda create -n myenv python=3.10
conda activate myenv

pip install -r requirements.txt

winget install ImageMagick.ImageMagick
brew install imagemagick

# 附加安装命令
#python -m spacy download en_core_web_sm  # 英文NLP模型
#python -m nltk.downloader punkt          # 分词数据

#brew tap oh-my-home/fonts
#brew install otf-source-han-sans-sc  # Install Source Han Sans - Language Specific OTFs Simplified Chinese.
#brew install otf-source-han-sans-tc  # Install Source Han Sans - Language Specific OTFs Traditional Chinese — Taiwan.
#brew install otf-source-han-sans-kc  # Install Source Han Sans - Language Specific OTFs Traditional Chinese — Hong Kong.
#brew install otf-source-han-sans-j  # Install Source Han Sans - Language Specific OTFs Japanese..
#brew install otf-source-han-sans-k  # Install Source Han Sans - Language Specific OTFs Korean.
#brew install ttc-source-han-serif  # Install Static Super OTC of Source Han Serif.
#brew install otf-source-han-serif-sc  # Install Source Han Serif - Language Specific OTFs Simplified Chinese.
#brew install otf-source-han-serif-tc  # Install Source Han Serif - Language Specific OTFs Traditional Chinese — Taiwan.
#brew install otf-source-han-serif-kc  # Install Source Han Serif - Language Specific

# 所有系统通用
#fc-cache -f -v

# 验证字体安装
#fc-list :lang=zh | grep -i "Noto Sans CJK"
# 应输出类似：/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc

# TODO: 
1. CALL DEEPSEEK to generate prompt, return in json format, use prompt, negative prompt, lora to improve pictures
2. Find a best matching model & lora combination