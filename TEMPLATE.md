# Template Usage Guide

## Getting Started with This Template

1. Click "Use this template" to create your new repository
2. Clone your new repository locally
3. Follow the setup instructions in README.md
4. Customize the project for your specific needs

## Customization Checklist

- [ ] Update `README.md` with your project name and description
- [ ] Modify `main.py` with your specific flow logic
- [ ] Add your nodes in `nodes.py`
- [ ] Configure environment variables in `.env`
- [ ] Update documentation in `docs/`
- [ ] Add your specific dependencies to `pyproject.toml`

## Template Structure

This template provides:
- ✅ PocketFlow node/flow architecture
- ✅ Python best practices setup
- ✅ Testing framework (pytest)
- ✅ Code quality tools (ruff, pyright)
- ✅ AI assistant integration (.cursorrules, CLAUDE.md)
- ✅ Comprehensive documentation

## Support

For issues with the template itself, please open an issue in the template repository.
For issues with your specific project, customize this section with your own support information. 

## i have a python project in cursor ide with a PRD. I want to use https://github.com/eyaltoledano/claude-task-master to create tasks that will guide the code development. i plan on using task master at the command line:

Installation

# Install globally

npm install -g task-master-ai

# OR install locally within your project

npm install task-master-ai

Initialize a new project

# If installed globally

task-master init

# If installed locally

npx task-master init

**PRD Parsing Command:**

```bash
task-master parse-prd scripts/prd.md --output-format markdown --ai-model claude-3.7-sonnet
```

Key parameters:

- `--dependency-graph` (visualize task relationships)
- `--complexity-threshold 5` (auto-split complex tasks)
- `--context @Codebase` (analyze existing

<div style="text-align: center">⁂</div>

[^1]: claude-task-master

[^2]: https://docs.task-master.dev

[^3]: https://www.youtube.com/watch?v=BvWzQ4W0QXA

[^4]: https://docs.anthropic.com/en/docs/claude-code/overview

[^5]: https://github.com/anthropics/claude-code-action

[^6]: https://www.reddit.com/r/ClaudeAI/comments/1kv4mfx/how_to_use_claude_code_in_vs_code_in_windows_via/

[^7]: https://samelogic.com/blog/claude-task-master-just-fixed-our-vibe-coding-workflow-heres-what-happened

[^8]: https://github.com/eighty9nine/task-master

[^9]: https://www.anthropic.com/engineering/claude-code-best-practices

[^10]: https://docs.anthropic.com/en/docs/claude-code/github-actions

[^11]: https://www.datacamp.com/tutorial/claude-opus-4-claude-code

[^12]: https://www.reddit.com/r/ClaudeAI/comments/1jlhg7g/task_master_how_i_solved_cursor_code_slop_and/

[^13]: https://www.reddit.com/r/ClaudeAI/comments/1ixave9/whats_claude_code/

[^14]: https://www.youtube.com/watch?v=H05Y-UJded0

[^15]: https://www.anthropic.com/claude-code

[^16]: https://github.com/eyaltoledano/claude-task-master/blob/main/docs/tutorial.md

[^17]: https://x.com/EyalToledano/status/1907636727247941640

[^18]: https://taskmaster.tv/tasks

[^19]: https://docs.anthropic.com/en/docs/claude-code/cli-usage

[^20]: https://github.com/anthropics/claude-code

