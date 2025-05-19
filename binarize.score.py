import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def sort(most_similar_pair, sorted_score_dict):
    # sorted_score_dict 딕셔너리와 인덱스를 기준으로 순위 정렬
    sorted_keys = list(sorted_score_dict.keys())  # 정렬된 Metric 리스트
    index_high = sorted_keys.index(most_similar_pair[0])
    index_low = sorted_keys.index(most_similar_pair[1])

    # 높은 순위와 낮은 순위를 구분하여 정렬
    if index_high > index_low:  # 높은 인덱스가 더 낮은 순위를 나타냄
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

    # 각 열을 0~1 사이로 정규화
    for col in df.columns[1:]:
        df[col] = normalize(df[col])
    df = df.fillna(df.mean(numeric_only=True))
    S = df.iloc[:, 4:].to_numpy()
    
    # Cosine 유사도 계산
    similarity_matrix = cosine_similarity(S)

    # Binary 유사도 행렬 생성 (임계값 0.9 사용)
    threshold = 0.9
    binary_similarity = (similarity_matrix >= threshold).astype(int)

    # 결과 출력
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)
    print("\nBinary Similarity Matrix:")
    print(binary_similarity)

    # Ranking score
    score_ranking = df.iloc[:, 1:].mean(axis=1).tolist()
    score_dict = dict(zip(df['Model'], score_ranking))
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))

    # UltraFeedback 받아오기
    ds = load_dataset("openbmb/UltraFeedback")
    data = pd.DataFrame(ds['train'])
    
    new_data = []

    for row_idx, row in tqdm(data.iterrows()):
        if len(row['completions'])<2:
            continue
        input_models = [model
                            for model in row['models'] 
                            if model != "ultralm-65b"
                        ]

        input_models_idx = [model_list.index(model) 
                            for model in row['models'] 
                            if model != "ultralm-65b"
                        ]
        # 부분 유사도 행렬 추출
        sub_matrix = similarity_matrix[np.ix_(input_models_idx, input_models_idx)]

        # 대각선 요소(자기 자신과의 유사도)를 제외한 유사도 값 찾기
        np.fill_diagonal(sub_matrix, -np.inf)  # 대각선 요소 제외
        max_idx = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)  # 최대값 인덱스
    
        # score_dict에서 높은 점수를 기준으로 정렬
        sorted_input_models = sorted(input_models, key=lambda x: score_dict[x], reverse=True)
        most_similar_pair = (sorted_input_models[0], sorted_input_models[1]) # sorted_input_models[:2]

        chosen_model = row['models'].index(most_similar_pair[0])
        rejected_model = row['models'].index(most_similar_pair[1])

        index_c = input_models.index(sorted_input_models[0])
        index_r = input_models.index(sorted_input_models[1])

        # sub_matrix에서 chosen의 행 가져오기
        rank_row = sub_matrix[index_c]

        # row의 내림차순 정렬 인덱스 계산
        sorted_indices = np.argsort(rank_row)[::-1]  # 내림차순 정렬

        # rejected 이 몇 번째로 가까운 값인지 찾기
        rank = np.where(sorted_indices == index_r)[0][0] + 1  # 0-based -> 1-based


        node = {}
        node['prompt'] = row['instruction']
        node['most_similar_pair'] = most_similar_pair
        chosen = [ { "content": row['instruction'], "role": "user" }, { "content": row['completions'][chosen_model]['response'], "role": "assistant" } ]
        rejected = [ { "content": row['instruction'], "role": "user" }, { "content": row['completions'][rejected_model]['response'], "role": "assistant" } ]
        node['chosen'] = chosen
        node['rejected'] = rejected
        node['similarity_rank'] = rank
        new_data.append(node)

        if row_idx % 10000==0 and row_idx!=0:
            new_data_df = pd.DataFrame(new_data)
            new_data_df.to_json('./model_aware/UltraFeedback.score.json', orient='records', indent=4)
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_json('./model_aware/UltraFeedback.score.json', orient='records', indent=4)
    new_data_df = pd.DataFrame(new_data)
    print(new_data_df['similarity_rank'].mean())

    frequency = new_data_df['similarity_rank'].value_counts().sort_index()

    # 막대 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.bar(frequency.index, frequency.values)
    plt.title("Frequency of Similarity Rank")
    plt.xlabel("Similarity Rank")
    plt.ylabel("Frequency")
    plt.xticks(frequency.index)  # X축 레이블 설정
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
