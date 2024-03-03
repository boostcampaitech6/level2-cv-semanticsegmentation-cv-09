# level-2 대회 : Hand Bone Image Segmentation 대회

## 팀 소개
| 이름 | 역할 |
| ---- | --- |
| [박상언](https://github.com/PSangEon) | SegNet, 시각화 |
| [지현동](https://github.com/tolfromj) | Git 관리, 증강기법 실험, U-Net 구현 |
| [오왕택](https://github.com/ohkingtaek) | PyTorch Lightning Code, Unet++, SwinV2-UperNet 구현 |
| [이동호](https://github.com/as9786) | Baseline Code, FCN, DilatedNet, DeepLabV3+ 구현 |
| [송지민](https://github.com/Remiing) | EDA, DeepLabV3 구현 |
| [이주헌](https://github.com/LeeJuheonT6138) | 증강기법 실험, Git 관리, Git 템플릿 작성 |

## 프로젝트 소개
<p align="center">
<img src="https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-09/assets/49676680/45cfbc1d-ad52-4b1e-8b96-899c0102f6d5">
</p>

뼈는 우리 몸의 구조화 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 중요하다. 예를 들어, 뼈의 형태나 위치가 변형된 것을 객체 분할 모형을 통해 확인할 경우, 해당 문제를 빠르게 해결할 수 있다. 데이터는 손가락, 손등 그리고 팔이 촬영된 X-ray 데이터 셋이다. 라벨은 총 29개로 각각의 라벨은 뼈 종류를 나타낸다. 우리는 사진 속에서 각 뼈의 종류들을 분할하는 작업을 수행해야 한다.

## 프로젝트 일정
프로젝트 전체 일정
- 02/05 10:00 ~ 02/21 19:00

프로젝트 세부 일정
- 02/05 ~ 02/07 강의 수강, 제공 데이터 및 코드 확인, BaseLine Code 작성
- 02/08 ~ 02/12 설날연휴, 휴식
- 02/13 ~ 02/16 데이터 살펴보기(EDA), 모형 실험
- 02/17 ~ 02/21 모형 실험, Git 정리

## 프로젝트 수행
- 데이터 살펴보기 & EDA : 클래스 불균형 확인, 겹치는 뼈 존재 확인
- 다양한 모형 실험 : FCN, U-Net, DeepLab, Swin Transformer 실험


## 프로젝트 결과
- 프로젝트 결과는 Public 7등, Private 6등이라는 결과를 얻었습니다.
![](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-09/assets/49676680/a44f7732-3a45-49d2-8318-e73d949c762f)
![](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-09/assets/49676680/ec68a2ba-7337-4a78-86e1-1b765d8ab39b)


## Wrap-Up Report

- [Wrap-Up Report](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-09/blob/main/docs/Semantic%20Seg_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(09%EC%A1%B0).pdf)

## File Tree

```bash
.
├── configs
│   ├── infer.yaml
│   └── train.yaml
├── docs
│   └── Semantic Seg_CV_팀 리포트(09조).pdf
├── dataset.py
├── infer.py
├── models.py
├── train.py
└── utils.py
```

| File(.py) | Description |
| --- | --- |
| dataset.py | Weighted Boxes Fusion 코드 |
| infer.py | train sh파일 |
| models.py | test sh파일 |
| train.py | train 코드 |
| utils.py | test 코드 |

## License
네이버 부스트캠프 AI Tech 교육용 데이터로 대회용 데이터임을 알려드립니다.
