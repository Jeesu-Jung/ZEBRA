import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def sort(most_similar_pair, score_dict):
    # sorted_score_dict 딕셔너리와 인덱스를 기준으로 순위 정렬
    index_high = score_dict[most_similar_pair[0]]
    index_low = score_dict[most_similar_pair[1]]

    # 높은 순위와 낮은 순위를 구분하여 정렬
    if index_high < index_low:  # 높은 점수가 더 높은 순위
        return (most_similar_pair[1], most_similar_pair[0])
    else:
        return (most_similar_pair[0], most_similar_pair[1])

if __name__=="__main__":
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
    print(S)
    # Cosine 유사도 계산
    similarity_matrix = cosine_similarity(S)

    # Binary 유사도 행렬 생성 (임계값 0.9 사용)
    threshold = 0.9
    binary_similarity = (similarity_matrix >= threshold).astype(int)
    # print(similarity_matrix)
    # print(binary_similarity)
    # 결과 출력
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)
    print("\nBinary Similarity Matrix:")
    print(binary_similarity)
    print(model_list)
    # Ranking score
    # 각 열을 0~1 사이로 정규화
    score_dict = dict(zip(df['Model'], score_ranking))
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))

    # UltraFeedback 받아오기
    ds = load_dataset("openbmb/UltraFeedback")
    data = pd.DataFrame(ds['train'])
    
    new_data = []

    for row_idx, row in tqdm(data.iterrows()):
        if len(row['completions'])<2:
            continue
        input_models = [model_list.index(model) 
                            for model in row['models'] 
                            if model != "ultralm-65b"
                        ]

        # 부분 유사도 행렬 추출
        sub_matrix = similarity_matrix[np.ix_(input_models, input_models)]

        # 대각선 요소(자기 자신과의 유사도)를 제외한 유사도 값 찾기
        np.fill_diagonal(sub_matrix, -np.inf)  # 대각선 요소 제외
        max_idx = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)  # 최대값 인덱스

        # 모델 인덱스 복원
        most_similar_models = (model_list[input_models[max_idx[0]]], model_list[input_models[max_idx[1]]])
        most_similar_pair = sort(most_similar_models, score_dict)

        chosen_model = row['models'].index(most_similar_pair[0])
        rejected_model = row['models'].index(most_similar_pair[1])

        node = {}
        node['prompt'] = row['instruction']
        node['most_similar_pair'] = most_similar_pair
        chosen = [ { "content": row['instruction'], "role": "user" }, { "content": row['completions'][chosen_model]['response'], "role": "assistant" } ]
        rejected = [ { "content": row['instruction'], "role": "user" }, { "content": row['completions'][rejected_model]['response'], "role": "assistant" } ]
        node['chosen'] = chosen
        node['rejected'] = rejected
        new_data.append(node)

        if row_idx % 10000==0 and row_idx!=0:
            new_data_df = pd.DataFrame(new_data)
            new_data_df.to_json('./model_aware/UltraFeedback.similarity.json', orient='records', indent=4)
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_json('./model_aware/UltraFeedback.similarity.json', orient='records', indent=4)


