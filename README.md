# Ethnicity-Classifier
📷 Paris 2024 Spectator Ethnicity Classifier (Deep Daiv. '24su Deep Learning Team)
(팀명: 디비딥러닝 - 이건, 이호균, 손보민, 황재령) 

## 프로젝트 배경
인공지능 연합 동아리 deep daiv. 2024 여름 기수 딥러닝 입문 <디비딥러닝> 팀의 첫 번째 프로젝트 : 파리 올림픽 관중석 인종 분류

## 사용 방법

1. 데이터 셋 준비
   - [UTKFACE 데이터셋](https://susanqq.github.io/UTKFace/) 준비
   - 전처리 된 데이터셋은 [dataset]()에서 확인 가능

2. 학습 파일 다운, 학습
  ```
python -m pip install -r requirements.txt
python train.py 
  ```

3.  추론
```
python test.py
```
