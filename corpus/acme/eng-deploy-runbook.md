---
id: acme/eng-deploy-runbook
title: Fleet Controller Deployment Runbook
acl: [engineering]
---
This runbook covers deploying the fleet controller, the service that assigns pick tasks to robots on the warehouse floor. Internal reference: CANARY-ACME-ENG-DEPLOY-4F1B.

Deploys go out through the staged pipeline: build, integration tests against the simulator, then a canary release to the Rotterdam pilot warehouse. The canary runs for at least 45 minutes and is promoted only if the task-assignment error rate stays below 0.5 percent and p99 assignment latency stays under 250 milliseconds.

Rollback: if either threshold is breached, run the rollback job from the pipeline, which redeploys the previous image tag in under four minutes. Never hotfix directly on the robots; every change must go through the pipeline.

Freeze windows: there are no deploys on Fridays after 14:00 local warehouse time, and none during the November peak season freeze, which runs from the second week of November until the first week of January. Emergency deploys during a freeze need sign-off from the on-call engineering manager.
