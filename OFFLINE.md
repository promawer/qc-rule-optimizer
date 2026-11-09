# 内网离线运行说明（Python 3.8）

这个压缩包是**自包含**的：自带 Python 3.8.10 运行时和全部依赖，内网机器不需要联网、不需要安装 Python/pip。

## 运行条件

- Windows 10 / 11 或 Windows Server 2016 及以上，**64 位**
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
  runtime\python\        自带 Python 3.8.10（64 位）及已安装依赖
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

- Python：3.8.10 embedding amd64
- FastAPI 0.115.6
- Uvicorn 0.30.6
- Pydantic 2.9.2（pydantic-core 2.23.4 cp38）
- HTTPX 0.27.2
- openpyxl 3.1.5
- python-multipart 0.0.20

全部 wheel 在 `wheelhouse` 中，且都声明支持 Python 3.8。

## 启动流程

`start.cmd` → `scripts\dev.ps1`：

1. 优先使用 `runtime\python\python.exe`（自带 3.8.10）；
2. 如果该目录被删除，则查找本机 `py -3.8` 或 `python`（必须 3.8 64 位）；
3. 回退模式下用 `.venv` + `wheelhouse` 离线安装，不访问网络；
4. 调用 `scripts\launch.py` 启动 Uvicorn，监听 `127.0.0.1:8090`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 双击后窗口一闪而过 | 在 cmd 中执行 `start.cmd`，查看报错；或看 `.run\app.log` |
| 端口 8090 被占用 | 先运行 `stop.cmd`，或关闭占用程序 |
| 浏览器没弹出 | 手动打开 <http://127.0.0.1:8090> |
| 提示缺少 runtime\python | 压缩包解压不完整，重新解压 |
| 安全软件拦截 | 将解压目录加入信任；服务只监听 127.0.0.1 |
| 想重置演示数据 | 先停止服务，删除 `data\store.json`，再启动 |

## 真实模型

右上角「模型设置」可切换到真实模型（DeepSeek 或 OpenAI 兼容网关）。需要内网能访问对应 API 地址，并在页面填写 API Key（只保存在进程内存）。
如果内网完全隔离，保持「模拟演示」即可，不会产生任何外部请求。

## 数据与备份

- 数据全部在 `data\store.json`，停止服务后直接复制即可备份；
- `.run\` 只放日志和 PID，可随时删除；
- 删除 `data\store.json` 会重置为演示数据。
