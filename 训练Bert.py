import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup,BertForSequenceClassification
from sklearn.model_selection import train_test_split
import json

# 数据预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        #定义编码器
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 从jsonl文件提取text和point
def load_data(file_path):
    texts=[]
    labels=[]
    with open(file_path,'r',encoding='utf-8') as F:
        for line in F:
            content=json.loads(line)
            texts.append(content["text"])
            labels.append(content["point"])
    return texts, labels


def train_model(model, train_loader, val_loader, device, epochs, learning_rate):
    model.to(device)
    optimizer =torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for i in range(epochs):
        model.to(device)
        model.train()
        total_loss = 0
        for data in train_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {i + 1}/{epochs}, Loss: {avg_loss}")


        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                correct+= (preds == labels).sum().item()
                total += labels.size(0)
        averageloss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Loss: {averageloss:.4f}, Accuracy: {accuracy:.4f}")

# 主函数
if __name__ == '__main__':
    file_path = "D:\\Users\\admin\\AppData\\Local\\Programs\\vscode代码\\.vscode\\__pycache__\\train_eval.jsonl"
    texts, labels = load_data(file_path)

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    print("dataset is completed")
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=16)
    print("dataloader is completed")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-chinese', num_labels=11)
    train_model(model, train_loader, val_loader,device, epochs=3, learning_rate=2e-5)
    torch.save(model.state_dict(),"D:\\Users\\admin\AppData\\Local\\Programs\\vscode代码\\.vscode\\__pycache__\\model_weight")
    model=torch.load("D:\\Users\\admin\AppData\\Local\\Programs\\vscode代码\\.vscode\\__pycache__\\model_weight")