import dataclasses
import unittest

from change_agent.terminal_review import (
    ALLOWED_TRIGGER_INPUTS,
    BackendCallIdentity,
    ReviewDecision,
    ReviewNextState,
    RoundTwoCallAudit,
    RoundTwoKind,
    RoundTwoObservation,
    RoundTwoOutcome,
    StageAttemptKind,
    TerminalReviewArm,
    TerminalReviewContractError,
    TerminalReviewCounters,
    TerminalReviewPolicy,
    TerminalReviewRequest,
    TerminalReviewScheduler,
    TerminalTrigger,
    ToolActionAuthorization,
    TRANSITION_TABLE,
    select_terminal_trigger,
    transition_for,
)


ROUND_ONE_PROMPT = "1" * 64
ROUND_TWO_PROMPT = "2" * 64
ALTERNATIVE_PROMPT = "3" * 64
TRIGGER_INPUT_FIELDS = tuple(sorted(ALLOWED_TRIGGER_INPUTS))


def counters(
    *,
    verifier_round_count=1,
    stage_initial_attempt_count=1,
    schema_repair_attempt_count=0,
    network_retry_attempt_count=0,
    tool_action_attempt_count=0,
    tool_failure_count=0,
    environment_step_count=0,
):
    return TerminalReviewCounters(
        verifier_round_count=verifier_round_count,
        stage_initial_attempt_count=stage_initial_attempt_count,
        schema_repair_attempt_count=schema_repair_attempt_count,
        network_retry_attempt_count=network_retry_attempt_count,
        tool_action_attempt_count=tool_action_attempt_count,
        tool_failure_count=tool_failure_count,
        environment_step_count=environment_step_count,
    )


def round_one_call(*, prompt=ROUND_ONE_PROMPT, region="r0"):
    return BackendCallIdentity("audit", region, prompt)


def call(
    *,
    prompt=ROUND_TWO_PROMPT,
    region="r0",
    attempt=StageAttemptKind.INITIAL,
    network_retries=0,
):
    return RoundTwoCallAudit(
        stage="audit",
        region_id=region,
        attempt_kind=attempt,
        prompt_sha256=prompt,
        raw_response='{"region_id":"r0"}',
        parsed_output={"region_id": region},
        network_retry_attempt_count=network_retries,
    )


def request(
    trigger,
    *,
    state_id="state-0",
    value=None,
    counts=None,
    call_identities=None,
    trigger_input_fields=TRIGGER_INPUT_FIELDS,
    round_two_already_used=False,
):
    return TerminalReviewRequest(
        accepted_state_id=state_id,
        round_one_value=value if value is not None else object(),
        trigger=trigger,
        counters=counts or counters(),
        trigger_input_fields=trigger_input_fields,
        round_one_call_identities=(
            call_identities
            if call_identities is not None
            else (round_one_call(),)
        ),
        round_two_already_used=round_two_already_used,
    )


def observation(outcome, *, value=None, calls=None, candidate_count=None):
    if outcome is RoundTwoOutcome.FINISH:
        retained = value if value is not None else object()
        count = 0
        completion = True
        schema_valid = True
    elif outcome is RoundTwoOutcome.UNIQUE_SAFE_PLAN:
        retained = value if value is not None else object()
        count = 1
        completion = False
        schema_valid = True
    elif outcome is RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN:
        retained = None
        count = 2 if candidate_count is None else candidate_count
        completion = False
        schema_valid = True
    elif outcome is RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR:
        retained = None
        count = 0
        completion = False
        schema_valid = False
    elif outcome is RoundTwoOutcome.BUDGET_EXHAUSTED:
        retained = None
        count = 0
        completion = None
        schema_valid = None
    else:
        retained = None
        count = 0
        completion = False
        schema_valid = True
    if candidate_count is not None:
        count = candidate_count
    if calls is None:
        if outcome is RoundTwoOutcome.BUDGET_EXHAUSTED:
            calls = ()
        elif outcome is RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR:
            calls = (
                call(),
                call(prompt=ALTERNATIVE_PROMPT, attempt=StageAttemptKind.REPAIR),
            )
        else:
            calls = (call(),)
    return RoundTwoObservation(
        outcome=outcome,
        value=retained,
        safe_plan_candidate_count=count,
        backend_calls=calls,
        completion_gate_passed=completion,
        schema_valid=schema_valid,
    )


class TerminalTransitionTableTest(unittest.TestCase):
    def test_transition_table_is_the_complete_two_by_seven_cross_product(self):
        expected_keys = {
            (trigger, outcome)
            for trigger in TerminalTrigger
            for outcome in RoundTwoOutcome
        }

        self.assertEqual(set(TRANSITION_TABLE), expected_keys)
        self.assertEqual(len(TRANSITION_TABLE), 14)

    def test_transition_table_matches_the_frozen_preregistration(self):
        expected = {
            (
                TerminalTrigger.PROVISIONAL_FINISH,
                RoundTwoOutcome.FINISH,
            ): (
                ReviewDecision.CONFIRMED_FINISH,
                ReviewNextState.TERMINATE_CONFIRMED_FINISH,
                ToolActionAuthorization.FORBIDDEN,
            ),
            (
                TerminalTrigger.PROVISIONAL_FINISH,
                RoundTwoOutcome.UNIQUE_SAFE_PLAN,
            ): (
                ReviewDecision.SAFE_PLAN_ENTERS_EXISTING_FLOW,
                ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
                ToolActionAuthorization.EXISTING_SAFETY_FLOW_REQUIRED,
            ),
            (
                TerminalTrigger.PROVISIONAL_FINISH,
                RoundTwoOutcome.PLAN_NULL,
            ): (
                ReviewDecision.ROUND_CONFLICT_NO_ACTION,
                ReviewNextState.TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW,
                ToolActionAuthorization.FORBIDDEN,
            ),
            (
                TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
                RoundTwoOutcome.FINISH,
            ): (
                ReviewDecision.ROUND_CONFLICT_NO_ACTION,
                ReviewNextState.TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW,
                ToolActionAuthorization.FORBIDDEN,
            ),
        }
        for trigger in TerminalTrigger:
            for outcome in RoundTwoOutcome:
                transition = transition_for(trigger, outcome)
                if (trigger, outcome) in expected:
                    self.assertEqual(
                        (
                            transition.decision,
                            transition.next_state,
                            transition.tool_action_authorization,
                        ),
                        expected[(trigger, outcome)],
                    )
                elif outcome is RoundTwoOutcome.UNIQUE_SAFE_PLAN:
                    self.assertEqual(
                        transition.decision,
                        ReviewDecision.SAFE_PLAN_ENTERS_EXISTING_FLOW,
                    )
                    self.assertEqual(
                        transition.next_state,
                        ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
                    )
                    self.assertEqual(
                        transition.tool_action_authorization,
                        ToolActionAuthorization.EXISTING_SAFETY_FLOW_REQUIRED,
                    )
                else:
                    self.assertEqual(
                        transition.decision,
                        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
                    )
                    self.assertEqual(
                        transition.next_state,
                        ReviewNextState.TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW,
                    )
                    self.assertEqual(
                        transition.tool_action_authorization,
                        ToolActionAuthorization.FORBIDDEN,
                    )

    def test_trigger_selector_accepts_only_the_two_frozen_predicates(self):
        self.assertIsNone(
            select_terminal_trigger(
                provisional_finish=False,
                plan_null_after_allowed_repair=False,
            )
        )
        self.assertEqual(
            select_terminal_trigger(
                provisional_finish=True,
                plan_null_after_allowed_repair=False,
            ),
            TerminalTrigger.PROVISIONAL_FINISH,
        )
        self.assertEqual(
            select_terminal_trigger(
                provisional_finish=False,
                plan_null_after_allowed_repair=True,
            ),
            TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        )
        with self.assertRaises(TerminalReviewContractError):
            select_terminal_trigger(
                provisional_finish=True,
                plan_null_after_allowed_repair=True,
            )


class TerminalReviewSchedulerTest(unittest.TestCase):
    def test_h0_is_identity_and_never_calls_round_two(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h0())
        sentinel = object()
        review_calls = []

        result = scheduler.resolve(
            request(
                TerminalTrigger.PROVISIONAL_FINISH,
                value=sentinel,
            ),
            lambda review_request: review_calls.append(review_request),
        )

        self.assertIs(result.value, sentinel)
        self.assertTrue(result.passthrough)
        self.assertIsNone(result.transition)
        self.assertEqual(review_calls, [])
        self.assertFalse(result.audit.round_two_invoked)
        self.assertEqual(result.audit.counters_before, result.audit.counters_after)

    def test_h0_identity_does_not_apply_h1_budget_validation(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h0())
        sentinel = object()
        review_calls = []

        result = scheduler.resolve(
            request(
                TerminalTrigger.PROVISIONAL_FINISH,
                value=sentinel,
                counts=counters(
                    verifier_round_count=2,
                    environment_step_count=4,
                ),
            ),
            lambda review_request: review_calls.append(review_request),
        )

        self.assertIs(result.value, sentinel)
        self.assertTrue(result.passthrough)
        self.assertEqual(review_calls, [])

    def test_h1_nontrigger_is_identity_and_never_calls_round_two(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        sentinel = object()
        review_calls = []

        result = scheduler.resolve(
            request(None, value=sentinel, call_identities=()),
            lambda review_request: review_calls.append(review_request),
        )

        self.assertIs(result.value, sentinel)
        self.assertTrue(result.passthrough)
        self.assertEqual(review_calls, [])
        self.assertFalse(result.audit.eligible)

    def test_every_frozen_transition_invokes_exactly_one_round_two(self):
        for trigger in TerminalTrigger:
            for outcome in RoundTwoOutcome:
                with self.subTest(trigger=trigger.value, outcome=outcome.value):
                    scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
                    review_calls = []
                    retained = object()

                    def observe(review_request):
                        review_calls.append(review_request)
                        return observation(outcome, value=retained)

                    result = scheduler.resolve(request(trigger), observe)
                    expected = transition_for(trigger, outcome)

                    self.assertEqual(len(review_calls), 1)
                    self.assertEqual(review_calls[0].round_index, 2)
                    self.assertFalse(review_calls[0].allow_round_three)
                    self.assertTrue(review_calls[0].gt_free)
                    self.assertEqual(result.transition, expected)
                    self.assertFalse(result.passthrough)
                    self.assertEqual(
                        result.audit.counters_after.verifier_round_count,
                        2,
                    )
                    if expected.next_state in {
                        ReviewNextState.TERMINATE_CONFIRMED_FINISH,
                        ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
                    }:
                        self.assertIs(result.value, retained)
                    else:
                        self.assertIsNone(result.value)
                    self.assertFalse(result.audit.tool_executed_by_scheduler)

    def test_round_kind_is_determined_only_by_the_frozen_trigger(self):
        seen = []
        for trigger, expected_kind in (
            (
                TerminalTrigger.PROVISIONAL_FINISH,
                RoundTwoKind.FINISH_CONFIRMATION,
            ),
            (
                TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
                RoundTwoKind.PLAN_NULL_ALTERNATIVE,
            ),
        ):
            scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

            def observe(review_request):
                seen.append(review_request)
                return observation(RoundTwoOutcome.PLAN_NULL)

            scheduler.resolve(request(trigger, state_id=trigger.value), observe)
            self.assertEqual(seen[-1].round_kind, expected_kind)

    def test_same_accepted_state_cannot_enter_round_two_twice(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        review_calls = []

        def observe(review_request):
            review_calls.append(review_request)
            return observation(RoundTwoOutcome.FINISH)

        first = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH), observe
        )
        second = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH), observe
        )

        self.assertEqual(len(review_calls), 1)
        self.assertEqual(
            first.transition.decision,
            ReviewDecision.CONFIRMED_FINISH,
        )
        self.assertEqual(
            second.transition.outcome,
            RoundTwoOutcome.BUDGET_EXHAUSTED,
        )
        self.assertFalse(second.audit.round_two_invoked)
        self.assertEqual(
            second.audit.contract_violation,
            "round_two_already_used_for_accepted_state",
        )
        self.assertIsNone(second.value)

    def test_reentrant_same_state_cannot_create_round_three(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        nested_observer_calls = []

        def observe(_):
            nested = scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda review_request: nested_observer_calls.append(review_request),
            )
            self.assertEqual(
                nested.audit.effective_outcome,
                RoundTwoOutcome.BUDGET_EXHAUSTED,
            )
            self.assertFalse(nested.audit.round_two_invoked)
            return observation(RoundTwoOutcome.FINISH)

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH), observe
        )

        self.assertEqual(result.transition.decision, ReviewDecision.CONFIRMED_FINISH)
        self.assertEqual(nested_observer_calls, [])

    def test_active_review_cannot_reset_away_its_exactly_once_token(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        def observe(_):
            with self.assertRaises(TerminalReviewContractError):
                scheduler.reset()
            return observation(RoundTwoOutcome.FINISH)

        scheduler.resolve(request(TerminalTrigger.PROVISIONAL_FINISH), observe)
        second = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(RoundTwoOutcome.FINISH),
        )

        self.assertFalse(second.audit.round_two_invoked)
        self.assertEqual(
            second.audit.effective_outcome,
            RoundTwoOutcome.BUDGET_EXHAUSTED,
        )

    def test_observer_failure_still_consumes_the_exactly_once_token(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        def fail(_):
            raise RuntimeError("observer failed")

        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                fail,
            )
        second = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(RoundTwoOutcome.FINISH),
        )

        self.assertFalse(second.audit.round_two_invoked)
        self.assertEqual(
            second.audit.contract_violation,
            "round_two_already_used_for_accepted_state",
        )

    def test_explicit_used_token_forbids_round_two(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        review_calls = []

        result = scheduler.resolve(
            request(
                TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
                round_two_already_used=True,
            ),
            lambda review_request: review_calls.append(review_request),
        )

        self.assertEqual(review_calls, [])
        self.assertEqual(
            result.transition.outcome,
            RoundTwoOutcome.BUDGET_EXHAUSTED,
        )
        self.assertIsNone(result.value)

    def test_reset_starts_a_new_episode_review_token_scope(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        review_calls = []

        def observe(review_request):
            review_calls.append(review_request)
            return observation(RoundTwoOutcome.PLAN_NULL)

        scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH), observe
        )
        scheduler.reset()
        scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH), observe
        )

        self.assertEqual(len(review_calls), 2)

    def test_round_one_must_be_counted_before_an_eligible_review(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(
                    TerminalTrigger.PROVISIONAL_FINISH,
                    counts=counters(verifier_round_count=0),
                ),
                lambda _: observation(RoundTwoOutcome.FINISH),
            )

    def test_round_two_request_has_no_tool_or_gt_execution_surface(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        captured = []

        scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda review_request: (
                captured.append(review_request)
                or observation(RoundTwoOutcome.FINISH)
            ),
        )

        fields = {item.name for item in dataclasses.fields(captured[0])}
        self.assertNotIn("tool", fields)
        self.assertNotIn("executor", fields)
        self.assertNotIn("gt_mask", fields)
        self.assertNotIn("offline_metrics", fields)
        self.assertFalse(captured[0].allow_round_three)


class TerminalReviewSafetyAndBudgetTest(unittest.TestCase):
    def test_forbidden_or_unknown_trigger_input_is_rejected(self):
        for field in (
            "gt_mask",
            "offline_tp_fp_fn",
            "offline_iou_f1",
            "manual_failure_label",
            "historical_false_finish_identity",
            "unregistered_hint",
        ):
            with self.subTest(field=field):
                with self.assertRaises(TerminalReviewContractError):
                    request(
                        TerminalTrigger.PROVISIONAL_FINISH,
                        trigger_input_fields=(field,),
                    )

    def test_gt_free_flags_are_mandatory(self):
        with self.assertRaises(TerminalReviewContractError):
            dataclasses.replace(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                gt_free=False,
            )
        with self.assertRaises(TerminalReviewContractError):
            dataclasses.replace(
                observation(RoundTwoOutcome.PLAN_NULL),
                gt_free=False,
            )

    def test_round_two_initial_prompt_or_region_must_be_independent(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        duplicate = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(call(prompt=ROUND_ONE_PROMPT, region="r0"),),
        )

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda _: duplicate,
            )

    def test_round_two_cannot_reuse_a_round_one_repair_prompt_identity(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        duplicate_repair = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(call(prompt=ALTERNATIVE_PROMPT, region="r0"),),
        )

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(
                    TerminalTrigger.PROVISIONAL_FINISH,
                    call_identities=(
                        round_one_call(prompt=ROUND_ONE_PROMPT, region="r0"),
                        round_one_call(prompt=ALTERNATIVE_PROMPT, region="r0"),
                    ),
                ),
                lambda _: duplicate_repair,
            )

    def test_stage_change_alone_does_not_make_duplicate_review_independent(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        duplicate = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(
                RoundTwoCallAudit(
                    stage="diagnosis",
                    region_id="r0",
                    attempt_kind=StageAttemptKind.INITIAL,
                    prompt_sha256=ROUND_ONE_PROMPT,
                    raw_response='{"region_id":"r0"}',
                    parsed_output={"region_id": "r0"},
                ),
            ),
        )

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda _: duplicate,
            )

    def test_budget_exhausted_with_a_call_still_requires_independence(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        duplicate = observation(
            RoundTwoOutcome.BUDGET_EXHAUSTED,
            calls=(call(prompt=ROUND_ONE_PROMPT, region="r0"),),
        )

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda _: duplicate,
            )

    def test_budget_exhausted_retains_independent_calls_and_their_counters(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(
                RoundTwoOutcome.BUDGET_EXHAUSTED,
                calls=(call(),),
            ),
        )

        self.assertEqual(
            result.transition.decision,
            ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        )
        self.assertTrue(result.audit.nonduplicate_prompt_or_region)
        self.assertEqual(len(result.audit.backend_calls), 1)
        self.assertEqual(
            result.audit.counters_after.stage_initial_attempt_count,
            result.audit.counters_before.stage_initial_attempt_count + 1,
        )

    def test_budget_exhausted_without_calls_does_not_claim_independence(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(RoundTwoOutcome.BUDGET_EXHAUSTED),
        )

        self.assertIsNone(result.audit.nonduplicate_prompt_or_region)
        self.assertEqual(result.audit.backend_calls, ())

    def test_new_prompt_on_same_region_is_an_independent_review(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(
                RoundTwoOutcome.PLAN_NULL,
                calls=(call(prompt=ROUND_TWO_PROMPT, region="r0"),),
            ),
        )

        self.assertTrue(result.audit.nonduplicate_prompt_or_region)

    def test_new_environment_ordered_region_is_an_independent_review(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR),
            lambda _: observation(
                RoundTwoOutcome.UNRESOLVED,
                calls=(call(prompt=ROUND_ONE_PROMPT, region="r1"),),
            ),
        )

        self.assertTrue(result.audit.nonduplicate_prompt_or_region)

    def test_hidden_state_or_raw_response_inheritance_is_rejected(self):
        base = observation(RoundTwoOutcome.PLAN_NULL)
        for update in (
            {"fresh_backend_context": False},
            {"inherits_hidden_state": True},
            {"copied_round_one_raw_response": True},
        ):
            with self.subTest(update=update):
                with self.assertRaises(TerminalReviewContractError):
                    dataclasses.replace(base, **update)

    def test_stage_attempt_budget_is_initial_plus_at_most_one_repair(self):
        excessive = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(
                call(),
                call(prompt="3" * 64, attempt=StageAttemptKind.REPAIR),
                call(prompt="4" * 64, attempt=StageAttemptKind.REPAIR),
            ),
        )
        repair_first = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(call(attempt=StageAttemptKind.REPAIR),),
        )
        for invalid in (excessive, repair_first):
            with self.subTest(calls=invalid.backend_calls):
                scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
                with self.assertRaises(TerminalReviewContractError):
                    scheduler.resolve(
                        request(TerminalTrigger.PROVISIONAL_FINISH),
                        lambda _, value=invalid: value,
                    )

    def test_schema_invalid_outcome_requires_recorded_allowed_repair(self):
        invalid = observation(
            RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR,
            calls=(call(),),
        )
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda _: invalid,
            )

    def test_local_round_two_network_retry_limit_is_zero(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        retried = observation(
            RoundTwoOutcome.PLAN_NULL,
            calls=(call(network_retries=1),),
        )

        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(TerminalTrigger.PROVISIONAL_FINISH),
                lambda _: retried,
            )

    def test_round_two_advances_only_verifier_and_stage_counters(self):
        before = counters(
            stage_initial_attempt_count=5,
            schema_repair_attempt_count=2,
            tool_action_attempt_count=2,
            tool_failure_count=1,
            environment_step_count=1,
        )
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(
                TerminalTrigger.PROVISIONAL_FINISH,
                counts=before,
            ),
            lambda _: observation(
                RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR
            ),
        )
        after = result.audit.counters_after

        self.assertEqual(after.verifier_round_count, 2)
        self.assertEqual(after.stage_initial_attempt_count, 6)
        self.assertEqual(after.schema_repair_attempt_count, 3)
        self.assertEqual(after.network_retry_attempt_count, 0)
        self.assertEqual(after.tool_action_attempt_count, 2)
        self.assertEqual(after.tool_failure_count, 1)
        self.assertEqual(after.environment_step_count, 1)

    def test_environment_budget_discards_unique_plan_before_existing_flow(self):
        safe_plan = object()
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(
                TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
                counts=counters(environment_step_count=3),
            ),
            lambda _: observation(
                RoundTwoOutcome.UNIQUE_SAFE_PLAN,
                value=safe_plan,
            ),
        )

        self.assertEqual(
            result.audit.observed_outcome,
            RoundTwoOutcome.UNIQUE_SAFE_PLAN,
        )
        self.assertEqual(
            result.audit.effective_outcome,
            RoundTwoOutcome.BUDGET_EXHAUSTED,
        )
        self.assertEqual(
            result.transition.tool_action_authorization,
            ToolActionAuthorization.FORBIDDEN,
        )
        self.assertIsNone(result.value)
        self.assertFalse(result.audit.safe_plan_returned_to_existing_flow)

    def test_only_unique_safe_plan_can_return_existing_flow_value(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        safe_plan = object()

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(
                RoundTwoOutcome.UNIQUE_SAFE_PLAN,
                value=safe_plan,
            ),
        )

        self.assertIs(result.value, safe_plan)
        self.assertEqual(
            result.transition.tool_action_authorization,
            ToolActionAuthorization.EXISTING_SAFETY_FLOW_REQUIRED,
        )
        self.assertTrue(result.audit.safe_plan_returned_to_existing_flow)
        self.assertFalse(result.audit.tool_executed_by_scheduler)

    def test_unique_safe_plan_requires_exactly_one_value(self):
        with self.assertRaises(TerminalReviewContractError):
            RoundTwoObservation(
                outcome=RoundTwoOutcome.UNIQUE_SAFE_PLAN,
                value=None,
                safe_plan_candidate_count=1,
                backend_calls=(call(),),
                completion_gate_passed=False,
                schema_valid=True,
            )
        with self.assertRaises(TerminalReviewContractError):
            RoundTwoObservation(
                outcome=RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
                value=object(),
                safe_plan_candidate_count=2,
                backend_calls=(call(),),
                completion_gate_passed=False,
                schema_valid=True,
            )

    def test_single_candidate_can_remain_unsafe_or_conflicted_without_a_value(self):
        for outcome in (
            RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
            RoundTwoOutcome.UNRESOLVED,
            RoundTwoOutcome.PLAN_NULL,
        ):
            with self.subTest(outcome=outcome.value):
                reviewed = observation(outcome, candidate_count=1)
                self.assertEqual(reviewed.safe_plan_candidate_count, 1)
                self.assertIsNone(reviewed.value)

    def test_policy_and_counter_drift_fail_closed(self):
        with self.assertRaises(TerminalReviewContractError):
            dataclasses.replace(
                TerminalReviewPolicy.h1(),
                max_environment_steps=4,
            )
        with self.assertRaises(TerminalReviewContractError):
            counters(tool_action_attempt_count=0, tool_failure_count=1)
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())
        with self.assertRaises(TerminalReviewContractError):
            scheduler.resolve(
                request(
                    TerminalTrigger.PROVISIONAL_FINISH,
                    counts=counters(environment_step_count=4),
                ),
                lambda _: observation(RoundTwoOutcome.FINISH),
            )

    def test_audit_is_json_ready_and_retains_required_budget_and_gt_fields(self):
        scheduler = TerminalReviewScheduler(TerminalReviewPolicy.h1())

        result = scheduler.resolve(
            request(TerminalTrigger.PROVISIONAL_FINISH),
            lambda _: observation(RoundTwoOutcome.FINISH),
        )
        payload = result.audit.to_dict()

        self.assertTrue(payload["gt_free"])
        self.assertEqual(payload["forbidden_trigger_inputs_present"], [])
        self.assertEqual(payload["round_index"], 2)
        self.assertEqual(payload["round_two_count"], 1)
        self.assertFalse(payload["round_three_allowed"])
        self.assertFalse(payload["inherits_hidden_state"])
        self.assertFalse(payload["copied_round_one_raw_response"])
        self.assertFalse(payload["tool_executed_by_scheduler"])
        self.assertEqual(
            payload["counter_scope"]["verifier_round_count"],
            "accepted_environment_state",
        )
        self.assertEqual(
            payload["counter_scope"]["tool_action_attempt_count"],
            "terminal_review_scheduler_only",
        )
        self.assertEqual(
            payload["counter_scope"]["environment_step_count"],
            "episode_cumulative_snapshot",
        )
        self.assertEqual(
            set(payload["counters_after"]),
            {
                "verifier_round_count",
                "stage_initial_attempt_count",
                "schema_repair_attempt_count",
                "network_retry_attempt_count",
                "tool_action_attempt_count",
                "tool_failure_count",
                "environment_step_count",
            },
        )
        self.assertEqual(
            payload["backend_calls"][0]["raw_response"],
            '{"region_id":"r0"}',
        )

    def test_policy_ids_are_frozen_per_arm(self):
        h0 = TerminalReviewPolicy.h0()
        h1 = TerminalReviewPolicy.h1()

        self.assertEqual(h0.arm, TerminalReviewArm.H0)
        self.assertFalse(h0.terminal_review)
        self.assertEqual(h0.max_verifier_rounds, 1)
        self.assertEqual(
            h0.harness_protocol_id,
            "qwen3vl_harness_h0_legacy_single_round_v1",
        )
        self.assertEqual(h1.arm, TerminalReviewArm.H1)
        self.assertTrue(h1.terminal_review)
        self.assertEqual(h1.max_verifier_rounds, 2)
        self.assertEqual(
            h1.harness_protocol_id,
            "qwen3vl_harness_h1_gtfree_terminal_review_v1",
        )


if __name__ == "__main__":
    unittest.main()
