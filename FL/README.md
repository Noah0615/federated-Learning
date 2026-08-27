# FL Challenge Telemetry

단일 GPU 머신에서 Docker Compose로 실행하는 Flower Deployment Runtime 미니 플릿입니다.
M0는 MNIST FedAvg 10라운드와 컨테이너 CPU/메모리 텔레메트리 확보를 목표로 하고,
M1 이후 `DeltaFedAvg`와 `attackers/precompute.py`를 단계적으로 활성화합니다.

## 빠른 시작

전제 조건은 Linux, Docker Compose V2, 그리고 호스트에 `flwr==1.34.0` CLI가 설치되어
있는 것입니다. GPU가 없어도 `gpu` 프로필을 제외하면 CPU-only M0를 실행할 수 있습니다.

```bash
python -m pip install "flwr==1.34.0"
docker compose up --build -d
docker compose --profile monitoring up -d
flwr run app local-deployment --stream
```

Flower CLI 설정 파일(`flwr config list`로 위치 확인)에 다음을 추가합니다.

```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

GPU와 DCGM을 사용할 때는 NVIDIA Container Toolkit이 설치된 Linux 호스트에서 다음처럼
실행합니다.

```bash
NUM_PARTITIONS=6 docker compose --profile gpu --profile monitoring up --build -d
```

Grafana는 [http://127.0.0.1:3000](http://127.0.0.1:3000), Prometheus는
[http://127.0.0.1:9090](http://127.0.0.1:9090)에서 확인할 수 있습니다.

## 구조

```text
app/challenge/       ServerApp, ClientApp, 모델, 전략, 기록기
app/attackers/       동일 ClientApp 인터페이스를 쓰는 공격 로직
monitoring/           Prometheus 설정
notebooks/            오프라인 탐지 분석 자리
results/              run별 parquet와 ground-truth manifest
docker/               Flower SuperExec 이미지
```

`NODE_IDENTITY`는 Compose가 부여하고, Flower의 연결별 node id와 함께 서버 로그 및
`rounds.parquet`에 기록합니다. `delta`는 서버의 `results/<run_id>/private/` 아래에만
덤프되며 Flower 메시지로 전송하지 않습니다.

## 단계별 실행

기본값은 M0를 위해 `delta-sigma=0.0`입니다. M1 실험은 다음처럼 실행합니다.

```bash
flwr run app local-deployment --stream \
  --run-config "delta-sigma=0.05 num-server-rounds=10"
```

공격 노드는 `compose.yaml`의 `superexec-evil` 서비스와 동일한 `ClientApp`을 사용하며,
`ATTACK_MODE=precompute` 환경 변수로 행동만 교체합니다. TLS, 온라인 탐지, Kubernetes,
멀티 머신 확장은 M3 산출물 이후의 범위입니다.

기본 스택은 M0용 정상 노드 5개입니다. 사전계산 공격 노드를 추가하려면 파티션 수를
6으로 맞추고 `attack` 프로필을 켭니다.

```bash
NUM_PARTITIONS=6 docker compose --profile attack up -d
```
