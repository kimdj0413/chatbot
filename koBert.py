import re
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

# CSV 파일 로드
train_data = pd.read_csv('C:/transformer_home/chatbot/dialogueData.csv')  # 파일명은 실제 파일 경로로 변경하세요

# KoBERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

# 최대 길이 설정
max_length = 128  # 원하는 최대 길이로 설정

# 데이터 전처리 및 토큰화
questions = []
answers = []

for sentence in train_data['req']:
    # 전처리
    sentence = re.sub(r'[^가-힣a-zA-Z0-9~?!,. ]', '', sentence)
    sentence = re.sub(r'([?.!,])', r" \1", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

for sentence in train_data['res']:
    if not isinstance(sentence, str):
        answers.append("문자열이 아닌 데이터 입니다")
        continue
    # 전처리
    sentence = re.sub(r'[^가-힣a-zA-Z0-9~?!,. ]', '', sentence)
    sentence = re.sub(r'([?.!,])', r" \1", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

# 토큰화
encoded_questions = tokenizer(
    questions, 
    padding='max_length', 
    truncation=True, 
    max_length=max_length, 
    return_tensors='pt'
)

encoded_answers = tokenizer(
    answers, 
    padding='max_length', 
    truncation=True, 
    max_length=max_length, 
    return_tensors='pt'
)

# 데이터셋 준비
labels = torch.tensor(range(len(answers)))  # 각 질문에 대한 인덱스 레이블
dataset = TensorDataset(encoded_questions['input_ids'], encoded_questions['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KoBERT 모델 로드
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=len(answers)).to(device)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# 학습 루프
model.train()
num_epochs = 3  # 에폭 수 설정
total_steps = len(dataloader) * num_epochs  # 총 스텝 수
print_interval = total_steps // 100  # 1%마다 출력

for epoch in range(num_epochs):
    total_loss = 0  # 에폭당 총 손실 초기화
    num_batches = 0  # 배치 수 초기화
    for batch_index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, label = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 총 손실에 추가
        num_batches += 1  # 배치 수 증가
        
        # 진행 상황 출력
        if (epoch * len(dataloader) + batch_index) % print_interval == 0:
            print(f'Progress: {((epoch * len(dataloader) + batch_index) / total_steps) * 100:.2f}%, '
                  f'Epoch: {epoch + 1}, Batch: {num_batches}, Loss: {loss.item():.4f}')
    
    # 에폭당 평균 손실 출력
    avg_loss = total_loss / num_batches
    print(f'Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}')

# 모델 저장
model.save_pretrained('my_kobert_chatbot_model')
tokenizer.save_pretrained('my_kobert_chatbot_model')



def generate_response(input_text):
    model.eval()
    with torch.no_grad():
        # 입력 텍스트 토큰화
        inputs = tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        ).to(device)
        
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        
        # 예측된 인덱스를 사용하여 응답 생성
        return answers[prediction.item()]

# 예시로 응답 생성
user_input = "안녕하세요!"
response = generate_response(user_input)
print(f"챗봇 응답: {response}")
