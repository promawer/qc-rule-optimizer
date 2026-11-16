# 内网离线运行说明（Python 3.8）

这个压缩包是**自包含**的：自带 Python 3.8.0（VS2017 构建）运行时和全部依赖，内网机器不需要联网、不需要安装 Python/pip。

## 运行条件

- Windows 7 SP1 / Server 2008 R2 SP1 及以上，**64 位**
- Windows 10 / 11 / Server 2016+ 同样适用
- 已针对 VC++2017 环境处理：不依赖 Windows 8+ 的 `pydantic-core`，Pydantic 使用纯 Python 实现
- 不需要管理员权限
- 不需要 Python、Node.js、npm、Maven
- 默认 mock 模式不访问网络；接真实 DeepSeek 时需要能访问配置的 API 地址

## 启动

1. 把 zip 拷贝到内网机器；
2. 解压到任意目录（路径可以有空格和中文）；
3. 双击 `qc-rule-optimizer\start.cmd`；
4. 浏览器打开 <http://127.0.0.1:8090>（通常会自动弹出）。

停止：双击 `stop.cmd`。

首次启动会创建 `data\store.json`（4 条演示规则 + 10 通示例通话）和 `.run\app.log`。
启动约 10～20 秒，日志会写到 `.run\app.log`。

## 目录说明

```
qc-rule-optimizer\
  runtime\python\        自带 Python 3.8.0（64 位，VS2017 构建）及已安装依赖
  wheelhouse\            全部离线 wheel（cp38 / win_amd64），用于回退安装
  app\                   后端代码
  static\                前端页面
  scripts\               启动/停止/自检脚本
  samples\               示例通话 CSV
  data\                  运行数据（首次启动创建）
  .run\                  日志和 PID（首次启动创建）
  start.cmd
  stop.cmd
  README.md
  OFFLINE.md
```

## 版本与依赖

- Python：3.8.0 embedding amd64（MSVC 19.16 / VS2017，匹配 VC++2017）
- FastAPI 0.115.6
- Uvicorn 0.30.6
- Pydantic 1.10.13（纯 Python，无 `pydantic-core`，兼容 Win7 / VC++2017）
- HTTPX 0.27.2
- openpyxl 3.1.5
- python-multipart 0.0.20

全部 wheel 在 `wheelhouse` 中，且都声明支持 Python 3.8。

## 启动流程

`start.cmd` → `scripts\dev.ps1`：

1. 优先使用 `runtime\python\python.exe`（自带 3.8.0）；
2. 如果该目录被删除，则查找本机 `py -3.8` 或 `python`（必须 3.8 64 位）；
3. 回退模式下用 `.venv` + `wheelhouse` 离线安装，不访问网络；
4. 调用 `scripts\launch.py` 启动 Uvicorn，监听 `127.0.0.1:8090`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 之前报 `DLL load failed while importing _pydantic_core` | 旧包使用了 Pydantic 2；本包已改为 Pydantic 1 纯 Python，不再有 `pydantic_core`。请使用 `qc-rule-optimizer-offline-win7-py38.zip` |
| 覆盖解压到旧目录后仍报旧错 | 删除旧目录下的 `.venv` 和 `wheelhouse`，或解压到一个全新目录再启动 |
| 双击后窗口一闪而过 | 在 cmd 中执行 `start.cmd`，查看报错；或看 `.run\app.log` |
| 端口 8090 被占用 | 先运行 `stop.cmd`，或关闭占用程序 |
| 浏览器没弹出 | 手动打开 <http://127.0.0.1:8090> |
| 提示缺少 runtime\python | 压缩包解压不完整，重新解压 |
| 自带 Python 无法启动 | 脚本会自动改用本机 Python 3.8 + `wheelhouse`，不会联网；确认本机是 3.8 64 位 |
| 安全软件拦截 | 将解压目录加入信任；服务只监听 127.0.0.1 |
| 想重置演示数据 | 先停止服务，删除 `data\store.json`，再启动 |

## 批量优化

「智能优化」页下方可以勾选多条规则或全选，按顺序逐条优化；任务在「任务中心」查看，每条规则的结果可以单独打开和应用。模型调用仍为单发串行。

## 规则文件导入

规则页左侧「导入」支持 CSV / JSON / XLSX，表头为：

```text
规则编号,规则名词,关键词,规则语义,规则内容的逻辑关系
```

按规则编号合并：已有编号更新，新编号新增，文件里没有的规则保持不动。关键词多个用 `|` 分隔；完整正则请加 `re:` 前缀。

## 真实模型

右上角「模型设置」可切换到真实模型（DeepSeek / OpenAI 兼容网关，或内网自定义接口）。

如果内网接口不是 OpenAI 格式，把「接口协议」改成 **自定义接口**，即可配置：

- 请求路径和全部请求表头（支持 `{{apiKey}}` 等变量）；
- JSON 请求体模板（`{{prompt}}`、`{{system}}`、`{{user}}`、`{{messages}}`、`{{model}}`、`{{requestId}}`、`{{max_tokens}}`、`{{temperature}}`）；
- 模板必须包含 `{{prompt}}`、`{{user}}` 或 `{{messages}}`，只写 `{{system}}` 不会上送通话文本；
- 返回内容路径，以及可选的违规布尔字段、理由字段路径。

配置后先点「测试连接」，成功后再跑优化或评测。自定义接口可不用 API Key，直接用固定表头鉴权。

如果内网接口只收一个 `question` 字段，点「单入参 question 模板」：

```json
{"question": "{{prompt}}"}
```

返回路径填 `result.finalAnswer` 即可。`{{prompt}}` 是项目自动拼好的完整提示词，包含判定/优化要求和 JSON 输出格式；接口本身不需要 system 字段，但模型必须返回项目需要的 JSON。

## 超时、重试与日志

「模型设置」→「AI 调用设置」：

- 单次超时：默认 90 秒，最大 300 秒；
- 失败重试：默认 3 次，0 表示不重试；
- 重试间隔：默认 1.5 秒，按次数递增；
- 调用间隔：默认 0.5 秒；
- 模型调用固定单发（并发 1），逐条评测，避免内网接口限流。

超时、连接失败、429/5xx 会自动重试；401/403/400 这类鉴权或配置错误不会反复重试。

页面会显示 AI 调用流程、调用统计、失败样本和完整运行日志。日志同时写入 `.run\jobs\<jobId>.log`，包含完整请求/响应体和状态/耗时（敏感表头已脱敏），可通过页面「下载完整日志」获取。

需要内网能访问对应 API 地址；如果内网完全隔离，保持「模拟演示」即可，不会产生任何外部请求。

## 数据与备份

- 数据全部在 `data\store.json`，停止服务后直接复制即可备份；
- `.run\` 只放日志和 PID，可随时删除；
- 删除 `data\store.json` 会重置为演示数据。
