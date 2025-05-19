import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def sort_by_score(models, score_dict):
    """모델 리스트를 성능 순으로 정렬"""
    return sorted(models, key=lambda model: score_dict[model], reverse=True)

if __name__ == "__main__":
    df = pd.read_csv('benchmark.csv', index_col=None)
    df = df.reset_index(drop=True)

    model_list = df['Model'].tolist()

    # % 제거 및 숫자 변환
    for col in df.columns[1:]:
        df[col] = df[col].replace("%", "", regex=True).astype(float)

    # 정규화 함수 (0~1 스케일)
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val)

    for col in df.columns[1:]:
        df[col] = normalize(df[col])

    score_ranking = df.iloc[:, 1:].mean(axis=1).tolist()
    df = df.fillna(df.mean(numeric_only=True))
    S = df.iloc[:, 1:].to_numpy()

    # Cosine 유사도 계산
    similarity_matrix = cosine_similarity(S)

    # Binary 유사도 행렬 생성 (임계값 0.9 사용)
    threshold = 0.9
    binary_similarity = (similarity_matrix >= threshold).astype(int)

    # Ranking score
    score_dict = dict(zip(df['Model'], score_ranking))
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))

    # UltraFeedback 받아오기
    # ds = load_dataset("openbmb/UltraFeedback")
    # data = pd.DataFrame(ds['train'])
    data = pd.read_json('./model_aware/UltraFeedback.baseline.score.json')
    model_list_without_ultralm = []
    for r_i, row in data.iterrows():
        if "ultralm-65b" in row['models']:
            row['models'].remove("ultralm-65b")
        model_list_without_ultralm.append(row['models'])
    data['models'] = model_list_without_ultralm
    
    new_data = []

    for row_idx, row in tqdm(data.iterrows()):
        if len(row['completions']) < 2:
            continue
        input_models = [model for model in row['models'] if model != "ultralm-65b"]

        # 기존 score 이용 시
        score_dict = {m:c for m, c in zip(row['models'], row['scores'])}

        # 후보 모델을 성능 순으로 정렬
        sorted_models = sort_by_score(input_models, score_dict)

        # 가장 성능이 높은 모델을 chosen으로 설정
        chosen_model_name = sorted_models[0]

        input_models = [model_list.index(model) 
                            for model in row['models'] 
                            if model != "ultralm-65b"
                        ]
        # 부분 유사도 행렬 추출
        sub_matrix = similarity_matrix[np.ix_(input_models, input_models)]
        np.fill_diagonal(sub_matrix, -np.inf)

        # chosen_model_name의 인덱스를 찾음
        chosen_index = row['models'].index(chosen_model_name)

        # chosen 모델의 유사도 벡터를 가져옴
        chosen_similarity_vector = sub_matrix[chosen_index, :]

        # 선택된 모델과 관련된 모델 중에서 가장 유사도가 높은 모델의 인덱스를 찾음
        rejected_index = np.argmax(chosen_similarity_vector)

        # 데이터 저장
        chosen_model = row['models'][chosen_index]
        rejected_model = row['models'][rejected_index]

        node = {}
        node['prompt'] = row['instruction']
        node['most_similar_pair'] = (chosen_model, rejected_model)
        chosen = [
            {"content": row['instruction'], "role": "user"},
            {"content": row['completions'][chosen_index]['response'], "role": "assistant"}
        ]
        rejected = [
            {"content": row['instruction'], "role": "user"},
            {"content": row['completions'][rejected_index]['response'], "role": "assistant"}
        ]
        node['chosen'] = chosen
        node['rejected'] = rejected
        new_data.append(node)

        if row_idx % 10000 == 0 and row_idx != 0:
            new_data_df = pd.DataFrame(new_data)
            new_data_df.to_json('./model_aware/UltraFeedback.combined.json', orient='records', indent=4)

    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_json('./model_aware/UltraFeedback.combined.json', orient='records', indent=4)
    print(new_data_df.head())
