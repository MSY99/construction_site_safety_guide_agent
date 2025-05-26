# Construction Site Safety Guide Agent

Construction Site Safety Guide Agent는 건설 현장에서의 안전을 보장하기 위한 다양한 법적 및 안전 사례를 제공하는 Python 기반의 에이전트입니다. 이 프로젝트는 법적 문서와 안전 사례를 분석하여 사용자에게 유용한 정보를 제공합니다.

## Installation

설치 방법
이 레포지토리를 클론합니다:

```bash
git clone https://github.com/MSY99/construction_site_safety_guide_agent.git
```

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

필요한 Python 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

환경 변수를 설정합니다:

.env 파일을 수정하여 필요한 환경 변수(GPT API key, etc.)를 설정합니다. 사용 방법 메인 애플리케이션 실행:

```bash
python src/main_app.py
```

주요 파일 설명
- src/agent_graph.py: 사용 X.
- src/config.json: MCP 도구 설정 정보를 담고 있는 json 파일.
- src/custom_mcps: Agentic RAG 도구 모듈 디렉토리.
- src/naive_rag_mcps: Naive RAG 도구 모듈 디렉토리.
- src/utils.py: Agent 응답을 stream으로 출력하도록 하는 utils가 있는 파일.