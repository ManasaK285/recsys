from __future__ import annotations
import asyncio, os, shutil, uuid
from pathlib import Path
from agents.base import get_llm
from agents.planner import Planner
from agents.repo_analyst import RepoAnalyst
from agents.coder import Coder
from agents.judge import Judge
from agents.meta_debugger import MetaDebugger
from verification.verifier import Verifier
from verification.requirements import check_requirements
from evaluation.failure_taxonomy import classify
from tracing.logger import TraceLogger
from storage.db import Database

class AgentForge:
    def __init__(self, db: Database|None=None, workspace_root: str|None=None):
        self.db=db or Database()
        self.workspace_root=workspace_root or os.getenv('AGENTFORGE_WORKSPACE_ROOT','artifacts/workspaces')
        Path(self.workspace_root).mkdir(parents=True,exist_ok=True)
        llm=get_llm(); self.planner=Planner(llm); self.meta=MetaDebugger(llm)
        self.repo=RepoAnalyst(); self.coder=Coder(); self.verifier=Verifier(); self.judge=Judge()

    async def run(self, task: str, repository: str, parallel_agents: int = 3) -> dict:
        run_id = uuid.uuid4().hex[:12]
        run_root = Path(self.workspace_root) / run_id
        run_root.mkdir(parents=True)

        trace = TraceLogger(str(run_root))
        self.db.save_event(
            run_id,
            'run_started',
            {'task': task},
        )

        spec = self.planner.plan(task)

        trace.log(
            'planning_complete',
            spec=spec.model_dump(),
        )

        self.db.save_event(
            run_id,
            'planning_complete',
            spec.model_dump(),
        )

        context = self.repo.analyze(repository)

        trace.log(
            'repository_analyzed',
            files=len(context['files']),
        )

        strategies = [
            'minimal',
            'robust',
            'test_first',
        ][:max(1, min(3, parallel_agents))]

        results = await self._attempt(
            run_id,
            run_root,
            repository,
            task,
            spec.model_dump(),
            context,
            strategies,
            trace,
        )

        verified = [
            r for r in results
            if r['judgment']['status'] == 'PASS'
        ]

        attempts = 1
        intervention = None

        # Trigger Meta-Debugger when any trajectory fails.
        failed = [
            r for r in results
            if r['judgment']['status'] != 'PASS'
        ]

        if failed:
            failures = [r['failure'] for r in failed]
            traces = [r['trace_summary'] for r in failed]

            intervention = self.meta.diagnose(
                failures,
                traces,
            )

            trace.log(
                'meta_debugger',
                intervention=intervention,
            )

            self.db.save_event(
                run_id,
                'meta_debugger',
                intervention,
            )

            attempts = 2

            results = await self._attempt(
                run_id,
                run_root,
                repository,
                task,
                spec.model_dump(),
                context,
                strategies,
                trace,
                intervention,
            )

            verified = [
                r for r in results
                if r['judgment']['status'] == 'PASS'
            ]

        status = 'PASS' if verified else 'FAIL'

        result = {
            'run_id': run_id,
            'status': status,
            'attempts': attempts,
            'spec': spec.model_dump(),
            'trajectories': results,
            'meta_debugger': intervention,
        }

        self.db.save_run(
            run_id,
            task,
            status,
            attempts,
            result,
        )

        trace.log(
            'run_finished',
            status=status,
            attempts=attempts,
        )

        return result
    async def _attempt(self, run_id, run_root, repository, task, spec, context, strategies, trace, intervention=None):
        async def one(strategy):
            workspace=run_root/f'{strategy}'
            trace.log('trajectory_started',strategy=strategy)
            result=self.coder.implement(repository,str(workspace),task,strategy,context,run_id,intervention)
            verification=self.verifier.run(str(workspace))
            req=check_requirements(str(workspace),task,spec)
            judgment=self.judge.evaluate(spec,verification,req)
            failure=None if judgment['status']=='PASS' else {'failure_class':classify(verification),'verification':verification}
            trace.log('trajectory_finished',strategy=strategy,status=judgment['status'],verification=verification)
            return {'strategy':strategy,'workspace':str(workspace),'verification':verification,'judgment':judgment,'failure':failure,'trace_summary':{'strategy':strategy,'status':judgment['status'],'failure':failure},'intervention':intervention}
        return await asyncio.gather(*(asyncio.to_thread(one_sync := (lambda s: None), s) for s in [])) if False else await asyncio.gather(*(asyncio.to_thread(self._one_sync, run_id, run_root, repository, task, spec, context, s, intervention) for s in strategies))

    def _one_sync(self, run_id, run_root, repository, task, spec, context, strategy, intervention):
        trace=TraceLogger(str(run_root)); existing = len(list(run_root.glob(strategy + '*')))
        workspace = run_root / f'{strategy}_attempt{existing + 1}'
        self.coder.implement(repository,str(workspace),task,strategy,context,run_id,intervention)
        verification=self.verifier.run(str(workspace)); req=check_requirements(str(workspace),task,spec); judgment=self.judge.evaluate(spec,verification,req)
        failure=None if judgment['status']=='PASS' else {'failure_class':classify(verification),'verification':verification}
        trace.log('trajectory_finished',strategy=strategy,status=judgment['status'],verification=verification)
        return {'strategy':strategy,'workspace':str(workspace),'verification':verification,'judgment':judgment,'failure':failure,'trace_summary':{'strategy':strategy,'status':judgment['status'],'failure':failure},'intervention':intervention}
