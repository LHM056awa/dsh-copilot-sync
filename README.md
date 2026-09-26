# dsh-copilot-sync — 把 dsh 模型列表离线同步到 VS Code Copilot BYOK

单向、本地、离线地把 `dsh/` 数据文件夹中的模型列表同步到 VS Code(Copilot
聊天 BYOK)维护的 `chatLanguageModels.json`。匹配规则:**同名端点**
(`dsh` 端点的 `name` 与目标端点的 `name` 一致),只写
`vendor == "customendpoint"` 条目。

- 不访问网络
- 不读取、不解析、不发送 API Key(`apiKey` 字段原样透传)
- 内置条目(`vendor == "copilot"` 等)字节级不变
- 失败时不破坏目标文件;重复运行幂等

## 安装

```bash
cd dsh-copilot-sync
pip install -e .
# 测试依赖
pip install -e ".[dev]"
```

需要 Python ≥ 3.10,依赖 `PyYAML`。

## 用法

```bash
python -m dsh_copilot_sync.cli \
  --dsh-dir <dsh 数据文件夹> \
  --config <chatLanguageModels.json 路径> \
  --all
```

也可用 `python -m dsh_copilot_sync`,或安装后的 `dsh-sync` 命令。

### 参数

| 参数 | 含义 |
| --- | --- |
| `--dsh-dir PATH` | dsh 数据文件夹路径(必须)。默认读取其中的 `settings.yaml.imported`(存在时),否则读取该目录下所有 `*.yaml`/`*.yml`/`*.json` 一级源文件(跳过 `credentials.yaml`、`sync.ffs_db`、`package.json`、点开头文件) |
| `--config PATH` | `chatLanguageModels.json` 路径(必须) |
| `--all` | 同步所有同名 `customendpoint`(与 `--provider` 二选一,必须指定其一) |
| `--provider NAME` | 只同步指定目标端点(可重复使用);名称在目标中不存在时报配置错误 |
| `--dry-run` | 只显示变更,不写文件 |
| `--no-delete` | 只新增;不删除模型、不删除 `settings` 项、不覆盖已有模型 |
| `--verbose` | 输出详细日志(未匹配端点、跳过的模型等) |
| `--version` | 显示版本 |

### 退出码

```text
0  成功
1  配置错误(源/目标文件缺失、JSON 无效、结构无法识别、重复端点名等)
2  部分端点失败(端点级错误已记录,可能已写入部分成功的内容)
3  写入目标文件失败
```

配置错误(退出码 1)时**绝不写目标文件**。

## dsh 数据源识别

适配层(`dsh_copilot_sync/dsh_source.py`)识别以下任一布局,混合亦可:

1. **期望的逻辑结构** — 顶层是端点对象列表(JSON 或 YAML):

   ```json
   [
     {
       "name": "EndpointA",
       "url": "https://api.example.com",
       "models": [
         { "id": "model-a", "name": "Model A", "url": "https://api.example.com" },
         { "id": "model-b" }
       ]
     }
   ]
   ```

2. **provider 分组文档** — 如 `settings.yaml.imported`:
   顶层 LLM 命名空间(形如 `llm-<provider>` 的键)的 `providers`
   子映射或直接挂 `models` 列表。

3. **插件 loader 文档** — 形如 `loader.yml`:顶层列表里各插件对象的
   `config.providers` 块。

字段归一化:

- 端点名:块的 `displayName` 优先;否则由 provider 键派生
  (去掉 `llm-` 前缀,按 `base_display_name` 规则:取最后一段路径、
  分隔符转空格、智能大写,如 `globex` → `Globex`、
  `llm-foo/provider-a` 键 `provider-a` → `Provider A`(以 `displayName` 为准时))。
- 模型 id 去重、保序;空 id、非字符串 id 跳过。
- 别名映射:`contextWindow` → `maxInputTokens`,`maxTokens` →
  `maxOutputTokens`,`inputModalities` 含 `image` → `vision: true`,
  含 `tools` → `toolCalling: true`。
- 块级 `baseURL` 作为该端点模型 `url` 的后备(去掉**末尾单个** `/v1` 段,
  例如 `https://host/api/v1` → `https://host/api`;不循环剥离,中间路径段保持不变)。
- 协议字段(`object`、`created`、`owned_by` 等)不会带入目标文件。

## 匹配规则

- 按 `name.trim()` 精确匹配,**大小写敏感**。
- 目标文件中同一端点名在 `customendpoint` 条目中出现多次:该端点报错
  并保持原样(非致命,其他端点继续同步)。
- 源端点名重复(跨文件或同文件内):**配置错误**(退出码 1)。
- 源中有、目标中没有同名端点:跳过并记录(`--verbose` 可见),**不创建新端点**。
  使用 `--provider` 时,"源中有、目标中没有"的记录与报告只覆盖被选中的端点;
  未被 `--provider` 选中的源端点不参与本次同步,也不计入未匹配列表(这是
  "只同步指定端点"的预期行为,而非缺陷)。
- 目标中有、源中没有同名端点:不修改。
- 非 `customendpoint` 条目(含 `vendor == "copilot"`)永远不修改。

## 同步规则(每个成功匹配的同名端点)

1. 源中有、目标中没有的模型 → 新增。
2. 目标中已有且源中仍有的模型 → 保留目标原对象(本地配置不被覆盖)。
3. 目标中有、源中没有的模型 → 默认删除,同时删除该端点顶层
   `settings` 中对应模型 id 的配置项;若 `settings` 因此变空,
   整个 `settings` 字段被删除。
4. `--no-delete`:只新增,不删除、不覆盖。
5. 源端点读取失败、模型列表为空、或没有任何有效模型 id → 跳过删除,
   目标端点原样保留,并记录错误(端点级,退出码 2)。
6. 新增模型字段仅限:`id`、`name`、`url`、`toolCalling`、`vision`、
   `maxInputTokens`、`maxOutputTokens`、`supportsReasoningEffort`。
   默认值 `toolCalling: true`、`vision: true`、`maxInputTokens: 1000000`、
   `maxOutputTokens: 384000`、`supportsReasoningEffort: ["max"]`;
   源提供时优先源值。
7. 新增模型 `url` 取值顺序(逐级回退,取第一个有值的级别):
   ① 源模型自身的 `url` → ② 源端点 `baseURL`(去掉末尾单个 `/v1` 后作为
   端点级 fallback)→ ③ 目标端点已有模型中**第一个**有 `url` 的模型值。
   三级都没有 → 跳过该模型并记录错误(不删除已有模型)。
   注意:③ 只取第一个,多 `url` 端点下不区分各模型归属的 url。
8. 新增模型 `name`:源值优先;否则按 `clm-sync` 的 `base_display_name`
   规则从模型 id 生成(如 `vendor-a/model-v4-flash` →
   `Model V4 Flash`)。

## 文件写入

- 原子写入:先写同目录临时文件(`.dsh-sync-*.tmp`),再 `os.replace` 替换。
- 写入前比较序列化内容(容忍 CRLF/LF 差异);无变化则完全不写。
- UTF-8、Tab 缩进、`ensure_ascii=False`,保持 VS Code 原有格式。
- 写入失败时清理临时文件,原文件保持完好(退出码 3)。

## 限制

- 只读 `--dsh-dir` 一级目录下的源文件(不含子目录递归)。
- 端点级错误(如某模型缺 url)时,该端点已有模型仍可能保留/新增,
  写入照常进行;只有全局配置错误(退出码 1)才完全不写。
- 同名匹配是精确字符串匹配:`EndpointA` 不会匹配 `endpointa`。
- 删除规则中"源端点读取失败"按端点粒度判断:只要该端点在源中存在且
  有 ≥1 个有效模型 id,删除即视为安全;源文件级解析失败是全局配置错误。

## 测试

```bash
python -m pytest
```

34 个用例覆盖规格中的 13 项必测场景(新增、保留本地配置、默认删除 +
`settings`、`--no-delete`、`--dry-run`、源读取失败、空模型列表、
非 customendpoint 不变、copilot 不变、源端点名重复报错、幂等二次运行、
原子写入同内容跳过、目标 JSON 无效/非数组),全部离线、不访问网络、
不处理真实 API Key。

## 工作区任务(可选)

在 `.vscode/tasks.json` 中添加:

```json
{
  "label": "Sync DSH Custom Endpoints",
  "type": "shell",
  "command": "python",
  "args": [
    "-m",
    "dsh_copilot_sync.cli",
    "--dsh-dir", "<dsh 数据文件夹路径>",
    "--config", "<chatLanguageModels.json 路径>",
    "--all"
  ],
  "group": { "kind": "build", "isDefault": true }
}
```
