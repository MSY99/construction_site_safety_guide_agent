# Construction Site Safety Guide Agent

Construction Site Safety Guide Agent는 건설 현장에서의 안전을 보장하기 위한 다양한 법적 및 안전 사례를 제공하는 Python 기반의 에이전트입니다. 이 프로젝트는 법적 문서와 안전 사례를 분석하여 사용자에게 유용한 정보를 제공합니다.

## 설치 요구 사항

- **Node.js, npm, npx 설치 필요**: 이 프로젝트는 Node.js 환경에서 실행됩니다. Node.js와 관련 도구들을 설치해야 합니다.

### Node.js 설치 방법

Node.js 설치 스크립트를 실행하여 Node.js를 설치합니다 (NodeSource LTS 사용):

```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y nodejs
```

설치 후 버전을 확인합니다:

```bash
node -v
npm -v
npx -v
```

설치 방법
이 레포지토리를 클론합니다:

```bash
git clone https://github.com/MSY99/construction_site_safety_guide_agent.git
```

필요한 Python 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

환경 변수를 설정합니다:

.env 파일을 수정하여 필요한 환경 변수를 설정합니다.
사용 방법
메인 애플리케이션 실행:

```bash
python src/main_app.py
```

에이전트는 다양한 법적 문서와 안전 사례를 분석하여 결과를 제공합니다.

주요 파일 설명
src/agent_graph.py: 에이전트 그래프 관련 코드.
src/config.json: 설정 정보를 담고 있는 파일.
src/custom_mcps: 사용자 정의 모듈 디렉토리.
src/naive_rag_mcps: 기본 모듈 디렉토리.
src/utils.py: 유틸리티 함수들이 포함된 파일.