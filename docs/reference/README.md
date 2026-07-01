# 参考文档

| 文件 | 内容 |
|------|------|
| [arch_yaml_schema.md](arch_yaml_schema.md) | arch YAML 完整字段表（baseline + 12 个细粒度 sweep 字段）+ 派生模板 + 规则 |
| [model_yaml_schema.md](model_yaml_schema.md) | model YAML schema（架构 + 推理超参 + GEMM ops + Vector ops + roofline calib）+ 完整 Qwen3 例 |
| [api.md](api.md) | prism Python 包对外 API（10 类入口 + dataclass 类型 + 异常处理）|
| [cli.md](cli.md) | 7 个 CLI 命令完整参数表 + 输出格式 + 退出码 |

**用途**：

- 加新 model / arch 派生 → [arch_yaml_schema.md](arch_yaml_schema.md) + [model_yaml_schema.md](model_yaml_schema.md)
- 集成到上层工作流 → [api.md](api.md)
- 命令行调用 → [cli.md](cli.md)

**进阶**：方法论原理见 [../methodology/](../methodology/)；典型工作流见 [../tutorials/](../tutorials/)。
