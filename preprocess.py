import pandas as pd
"""
df1 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/Talk_2024.8.6 14_04-1.txt', delimiter='\n')
df2 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/Talk_2024.8.6 14_04-2.txt', delimiter='\n')
df3 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/Talk_2024.8.6 14_04-3.txt', delimiter='\n')
df4 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/Talk_2024.8.6 14_04-4.txt', delimiter='\n')

df4 = df4.iloc[2:]
print(len(df4))
print(df4)
datePattern = r'^\d{4}년 \d{1,2}월 \d{1,2}일 (월|화|수|목|금|토|일)요일$'
mask = df4.apply(lambda row: row.astype(str).str.match(datePattern).any(), axis=1)
df4 = df4[~mask]
print(len(df4))
keywords = ['김동준 : ', '봄버맨 : ']
questions = []
answers = []
def remove_until_keyword(text):
    for keyword in keywords:
        if keyword in text:
            if keyword == "김동준 : ":
                questions.append(text.split(keyword, 1)[1])
                answers.append('')
            else:
                answers.append(text.split(keyword, 1)[1])
                questions.append('')
    return text

df4.iloc[:, 0] = df4.iloc[:, 0].apply(remove_until_keyword)

df = pd.DataFrame(list(zip(questions, answers)), columns=['questions','answers'])
remove_strings = ["이모티콘", "사진", "동영상","송금","봉투","정산"]
df = df[~df['questions'].apply(lambda x: any(r in x for r in remove_strings))]
df = df[~df['answers'].apply(lambda x: any(r in x for r in remove_strings))]
df.reset_index(drop=True, inplace=True)
for index, row in df.iterrows():
    if row['questions'].strip() == "" or row['answers'].strip() == "":
        if index + 1 < len(df):
            df.loc[index + 1, 'questions'] = df.loc[index, 'questions'] + " " + df.loc[index + 1, 'questions']
            df.loc[index + 1, 'answers'] = df.loc[index, 'answers'] + " " + df.loc[index + 1, 'answers']

df.replace(r'^\s*$', float('nan'), regex=True, inplace=True)

# NaN이 있는 행 삭제
df.dropna(how='any', inplace=True)
df['questions'] = df['questions'].str.lstrip()

# 'answers' 열의 맨 앞 공백 삭제 (필요한 경우)
df['answers'] = df['answers'].str.lstrip()
df.to_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel_4.csv',index=False)
"""
df1 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel_1.csv')
df2 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel_2.csv')
df3 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel_3.csv')
df4 = pd.read_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel_4.csv')
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

combined_df.to_csv('C:/transformer_home/Kakaotalk_Chat_봄버맨/parrerel.csv',index=False)