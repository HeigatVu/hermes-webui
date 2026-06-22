"""Static-source assertions for the #4662 profile-switch loading skeletons.

These don't execute JS — they assert the source wiring so the behaviour can't
silently regress:

  * switchToProfile() shows both skeletons up front (clears stale content),
    parallelizes the independent list+workspace refreshes, and restores real
    content on failure so a skeleton never strands.
  * renderSessionListFromCache() clears the skeleton-active flag on real render.
  * style.css defines the skeleton classes, the sheen + fade keyframes, the
    reduced-motion fallback, and dark-mode tokens.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
PANELS = (REPO_ROOT / "static" / "panels.js").read_text(encoding="utf-8")
SESSIONS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
WORKSPACE = (REPO_ROOT / "static" / "workspace.js").read_text(encoding="utf-8")
CSS = (REPO_ROOT / "static" / "style.css").read_text(encoding="utf-8")


def _switch_body() -> str:
    start = PANELS.index("async function switchToProfile(")
    # grab a generous slice (the function is long); next top-level function after it
    end = PANELS.index("function openProfileCreate(", start)
    return PANELS[start:end]


class TestSwitchWiring:
    def test_shows_session_skeleton_up_front(self):
        body = _switch_body()
        assert "showSessionListSkeleton()" in body
        # ...and before the awaited /api/profile/switch POST, so stale rows clear immediately
        assert body.index("showSessionListSkeleton()") < body.index("/api/profile/switch")

    def test_shows_workspace_skeleton_when_panel_open(self):
        body = _switch_body()
        assert "showWorkspaceTreeSkeleton()" in body
        assert "_workspaceVisibleAtStart" in body

    def test_workspace_refresh_after_switch_guard(self):
        # The workspace tree refresh (loadDir) must run AFTER the stale-switch
        # generation guard — loadDir paints the tree with only a session-id
        # check, and empty-session switches reuse the session id, so starting it
        # before the guard could let an older switch paint over a newer one
        # (Codex gate #4662). The skeletons shown up front already provide the
        # immediate cross-surface feedback.
        body = _switch_body()
        guard = "if (_switchGen !== _profileSwitchGeneration) return;"
        # In the non-sessionInProgress branch, the loadDir('.') call must come
        # after an occurrence of the guard.
        dir_idx = body.index("const dirLoad = loadDir('.');")
        guard_before = body.rfind(guard, 0, dir_idx)
        assert guard_before != -1, "loadDir('.') must be preceded by the switch-generation guard"

    def test_restores_real_content_on_failure(self):
        body = _switch_body()
        catch = body[body.index("} catch (e) {"):]
        assert "renderSessionListFromCache()" in catch, "failed switch must restore real list"

    def test_noop_self_switch_early_returns(self):
        # Opus gate #4662: a switch to the already-active profile must bail before
        # showing a skeleton (activateCurrentProfile() doesn't pre-check), else it
        # flashes skeleton→restore.
        body = _switch_body()
        head = body[: body.index("showSessionListSkeleton()")]
        assert "name === S.activeProfile" in head, "missing no-op self-switch early-return"
        assert "return;" in head

    def test_dismisses_rename_and_menu_before_skeleton(self):
        # Opus gate #4662: renderSessionListFromCache() early-returns while
        # _renamingSid / _sessionActionMenu is set — which would strand the
        # skeleton. switchToProfile must dismiss both before showing it.
        body = _switch_body()
        pre = body[: body.index("showSessionListSkeleton()")]
        assert "_renamingSid = null" in pre, "must clear inline-rename state before skeleton"
        assert "closeSessionActionMenu()" in pre, "must close row action menu before skeleton"

    def test_clears_workspace_skeleton_when_no_workspace(self):
        # Opus gate #4662 (blocker): if the new profile has no bound workspace the
        # real loadDir is skipped, so the up-front workspace skeleton must be
        # explicitly cleared or it strands forever.
        body = _switch_body()
        assert body.count("clearWorkspaceTreeSkeleton()") >= 2, (
            "both switch branches must clear a stranded workspace skeleton"
        )


class TestSessionsWiring:
    def test_skeleton_flag_cleared_on_real_render(self):
        # renderSessionListFromCache clears the skeleton-active flag when it
        # writes real rows (so a strand can't persist).
        idx = SESSIONS.index("function renderSessionListFromCache(")
        body = SESSIONS[idx: idx + 4000]
        assert "_sessionListSkeletonActive=false" in body.replace(" ", "")

    def test_builder_defines_groups_and_function(self):
        assert "const _SESSION_SKELETON_GROUPS" in SESSIONS
        assert "function showSessionListSkeleton(" in SESSIONS

    def test_skeleton_tears_down_virtual_scroll_state(self):
        # #4662 Codex gate: on long virtualized sidebars, leaving the
        # data-session-virtual-* window state + a queued scroll RAF active would
        # let _scheduleSessionVirtualizedRender() repaint the PREVIOUS profile's
        # cached rows over the skeleton. The builder must clear that state.
        idx = SESSIONS.index("function showSessionListSkeleton(")
        body = SESSIONS[idx: idx + 2000]
        assert "cancelAnimationFrame(_sessionVirtualScrollRaf)" in body
        assert "delete list.dataset.sessionVirtualTotal" in body
        assert "delete list.dataset.sessionVirtualStart" in body
        assert "delete list.dataset.sessionVirtualEnd" in body

    def test_virtual_render_guarded_by_skeleton_flag(self):
        # The virtual-scroll scheduler must bail while a skeleton is up.
        idx = SESSIONS.index("function _scheduleSessionVirtualizedRender(")
        body = SESSIONS[idx: idx + 600]
        assert "if(_sessionListSkeletonActive) return;" in body


class TestWorkspaceWiring:
    def test_builder_defined(self):
        assert "const _WS_SKELETON_ROWS" in WORKSPACE
        assert "function showWorkspaceTreeSkeleton(" in WORKSPACE

    def test_skeleton_clear_helper_defined(self):
        # The strand-clear helper must exist and only empty #fileTree when it
        # still holds a skeleton (so it can't clobber a real render).
        assert "function clearWorkspaceTreeSkeleton(" in WORKSPACE
        idx = WORKSPACE.index("function clearWorkspaceTreeSkeleton(")
        body = WORKSPACE[idx: idx + 400]
        assert ".skeleton-tree" in body, "clear helper must check for a skeleton before emptying"


class TestSkeletonCss:
    def test_core_classes_present(self):
        for cls in (".skeleton-list", ".skeleton-row", ".skeleton-bar",
                    ".skeleton-group-label", ".skeleton-tree", ".skeleton-tree-row",
                    ".skeleton-glyph"):
            assert cls in CSS, f"missing skeleton CSS class {cls}"

    def test_sheen_and_fade_keyframes(self):
        assert "@keyframes skeletonSheen" in CSS
        assert "@keyframes skeletonFadeIn" in CSS
        assert "animation:skeletonSheen" in CSS.replace(" ", "")

    def test_reduced_motion_disables_animation(self):
        # There must be a prefers-reduced-motion block that turns the skeleton
        # sheen animation off (accessibility contract).
        compact = CSS.replace(" ", "")
        assert "prefers-reduced-motion" in compact
        # The reduced-motion rule names .skeleton-bar and sets animation:none.
        rm_blocks = [b for b in compact.split("@media(prefers-reduced-motion:reduce){")
                     if ".skeleton-bar" in b[:400]]
        assert rm_blocks, "no reduced-motion block scoping .skeleton-bar"
        assert "animation:none" in rm_blocks[0][:400], "reduced-motion must disable the sheen"

    def test_theme_tokens_defined_for_light_and_dark(self):
        compact = CSS.replace(" ", "")
        assert "--skeleton-base:" in compact
        assert "--skeleton-sheen:" in compact
        assert ":root.dark{--skeleton-base:" in compact
