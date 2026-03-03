# MCP Server — Execution Explained

## 1. O que acontece ao executar `PYTHONPATH=. python -m mcp_server.server`

O Python carrega o módulo `mcp_server.server` como `__main__`. O código top-level executa **na importação** — antes de `uvicorn.run()`:

```
┌─────────────────────────────────────────────┐
│  BOOTSTRAP (top-level, síncrono)            │
│                                             │
│  1. SchemaRegistry()     → 10 tool schemas  │
│  2. ContextManager()     → load context.json│
│  3. ResourceRegistry()   → load *.json      │
│  4. ToolRegistry()       → empty            │
│  5. ExecutionEngine()    → load experiments  │
│  6. register_all_tools() → bind 10 funcs    │
│                                             │
│  SERVIDOR PRONTO                            │
├─────────────────────────────────────────────┤
│  uvicorn.run() → HTTP loop em 0.0.0.0:8000 │
└─────────────────────────────────────────────┘
```

## 2. Sequência Interna Detalhada

### 2.1 Bootstrap (antes de aceitar requests)

| Etapa | Componente | O que faz | Falha = |
|-------|-----------|-----------|---------|
| 1 | `SchemaRegistry()` | Carrega 10 definições JSON Schema | Crash imediato |
| 2 | `ContextManager(REGISTRY_PATH)` | Lê `registry/context.json`, restaura contextos salvos | Crash se JSON inválido |
| 3 | `ResourceRegistry(REGISTRY_PATH)` | Lê `models.json`, `metrics.json`, `datasets.json` | Crash se JSON inválido |
| 4 | `ToolRegistry(SchemaRegistry)` | Instancia registro vazio de ferramentas | Nunca falha |
| 5 | `ExecutionEngine(...)` | Lê `experiments.json`, restaura ledger | Crash se JSON inválido |
| 6 | `register_all_tools(tool_registry)` | Registra as 10 tool functions | Crash se schema não existe |

### 2.2 Evento de Startup (assíncrono)

O `lifespan` handler executa:
- **Startup**: Nenhuma ação adicional (estado já carregado no bootstrap síncrono)
- **Shutdown**: Safety net — componentes já persistem a cada operação

### 2.3 Aceitação de Requests

Uvicorn inicia o loop HTTP. Endpoints disponíveis:

| Endpoint | Método | Função |
|----------|--------|--------|
| `/health` | GET | Health check + contagem de tools/resources |
| `/tools/list` | POST | Lista 10 tools com schemas completos |
| `/tools/call` | POST | Executa tool com lifecycle completo |
| `/resources/list` | POST | Lista resources por tipo |
| `/resources/get` | POST | Busca resource por ID |
| `/experiments` | GET | Lista experiments com filtros |
| `/experiments/{id}` | GET | Detalhe de experiment |
| `/audit/physical` | GET | Verifica existência física de arquivos |
| `/audit/consistency` | GET | Verifica consistência do registry |

## 3. O que NÃO acontece automaticamente

| Ação | Requer chamada explícita |
|------|--------------------------|
| Carregamento de datasets | `tool.load_dataset` |
| Treinamento de modelos | `tool.run_baseline`, `tool.train_qsvm`, etc. |
| Comparação de modelos | `tool.compare_models` |
| Deploy IBM Quantum | `tool.deploy_ibm` + `IBM_QUANTUM_TOKEN` |
| PCA | Automático quando `feature_dim != n_qubits` |
| Criação de diretórios | Automático via `tool.initialize_project` |

## 4. Fluxo Completo de Execução de uma Tool

```
POST /tools/call { "tool_name": "tool.X", "arguments": {...} }
        │
        ▼
┌── ExecutionEngine.execute() ──────────────────────────┐
│                                                        │
│  1. LOCK: gera experiment_id único (UUID, anticolisão) │
│  2. Registra experiment como PENDING                   │
│  3. Persiste experiments.json (atômico)                │
│                                                        │
│  4. ToolRegistry.validate_and_check():                 │
│     ├── tool existe? → se não: ToolError               │
│     ├── input válido? → se não: SCHEMA_VALIDATION      │
│     └── preconditions? → se não: PRECONDITION_FAILED   │
│                                                        │
│  5. ContextManager.create_context():                   │
│     ├── seed=42, backend=aer_statevector               │
│     ├── dataset_hash se disponível                     │
│     └── Persiste context.json                          │
│                                                        │
│  6. experiment.status = RUNNING, persiste              │
│                                                        │
│  7. Executa tool function:                             │
│     ├── sync → run_in_executor                         │
│     └── async → await direto                           │
│                                                        │
│  8. Valida output contra schema:                       │
│     ├── OK → COMPLETED                                 │
│     └── warnings → COMPLETED_WITH_WARNINGS             │
│                                                        │
│  9. experiment.status = COMPLETED, persiste            │
│                                                        │
│  CATCH ToolError:                                      │
│     → status = FAILED                                  │
│     → lessons.md atualizado                            │
│     → retorna erro estruturado (sem crash)             │
│                                                        │
│  CATCH Exception:                                      │
│     → status = FAILED                                  │
│     → traceback registrado                             │
│     → lessons.md atualizado                            │
│     → retorna erro estruturado (sem crash)             │
│                                                        │
│  FINALLY:                                              │
│     → persiste experiments.json (sempre)               │
└────────────────────────────────────────────────────────┘
```

## 5. Garantias do Sistema

### 5.1 Persistência (Invariante 1)
- **Mecanismo**: Escritas atômicas (`tmp` + `rename`) em cada mutação
- **Teste**: Registrar resource → reiniciar → resource persiste
- **Risco mitigado**: Corrupção por crash mid-write (escrita atômica via tmp file)

### 5.2 Determinismo (Invariante 2)
- **Mecanismo**: Cada contexto grava `seed`, `dataset_hash`, `backend`, `model_version`
- **Teste**: Re-executar com mesmo seed → variância < 1%
- **Risco mitigado**: Resultados não reprodutíveis (seed fixo + hash de dataset)

### 5.3 Isolamento de Tools (Invariante 3/4)
- **Mecanismo**: Tools recebem context via parâmetros, nunca acessam outros tools
- **Teste**: Tool inexistente → erro estruturado, sem crash
- **Risco mitigado**: Cascata de falhas entre tools (isolamento total)

### 5.4 Autonomia sob Reinicialização (Invariante 5)
- **Mecanismo**: Todo estado em `registry/*.json`, carregado no bootstrap
- **Teste**: Parar servidor, reiniciar, verificar hash idêntico
- **Risco mitigado**: Perda de estado (persistência imediata, não lazy)

### 5.5 Reprodutibilidade Científica
- **Mecanismo**: SHA-256 de datasets + modelo + contexto versionado
- **Teste**: Hash de dataset antes/depois deve ser idêntico
- **Risco mitigado**: "It worked on my machine" (hash determinístico)

## 6. Riscos Estruturais e Mitigações

| Risco | Severidade | Mitigação |
|-------|-----------|-----------|
| JSON corrompido no `registry/` | Alta | Escritas atômicas via `tmp` + `rename` |
| Colisão de `experiment_id` | Média | UUID4 + 100 retries com verificação |
| Concorrência em escritas | Alta | `asyncio.Lock()` no ExecutionEngine |
| Dataset silenciosamente diferente | Alta | SHA-256 hash obrigatório |
| Tool crasha o servidor | Alta | Try/except em `execute()` — nunca propaga |
| Modelo sem hash | Média | Hash computado automaticamente no registro |
| IBM Quantum token expirado | Baixa | Verificação prévia via `check_ibm_token()` |
| Perda de lessons.md | Baixa | Write em `try/except` — falha silenciosa |
