/**
 * tests/traits.test.mjs — 특성 (§11.6, v1.10 ⑲)의 정본 계약.
 *
 * 커버:
 *   구슬   — 스테이지 보스 코어 격파 → 금색 구슬(kind 'trait') 이 자석으로 날아와 먹히면 traitQueue+1 · 그동안 STAGE_CLEAR 는 기다린다
 *            최종(테마 없음)은 구슬 없음
 *   드래프트 — offerCount 장 · 테마별 하나 우선, 모자라면 남은 것으로 채움 · 가진 특성만 제외(★ ㉑ 묶음 배타 없음 — 같은 테마 여러 개)
 *            · 5번째 보스에서도 3장 · 결정성 · applyCard 가 traitQueue 를 소비
 *   효과   — 재생 · 회수 · 보급 강화 · 격벽(피격 1회 무효 + 재충전) · 재기(치명상 → HP 1, 스테이지당 1회) · 장갑(정액 감산 + 하한)
 *            · 청정 · 보스 사냥꾼 · 처형(잔여 HP ≤ 25%) · 스탠스 공명(탄 소거 + 무적) · 전환 가속(발사 주기 창) · 전환 회수(픽업 자석)
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy, spawnEnemyBullet, spawnPickup, applyTrait, recomputeTraitFx, recomputeEff } from '../src/core/state.js';
import { step, makeInput, TICK_DT, killEnemy, applyHit } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook, spawnBoss } from '../src/core/boss.js';
import { initRun, tickRun, advanceStage, applyStageClearHeal, traitPickupAlive, PHASE } from '../src/core/stage.js';
import { buildTraitDraft, applyCard } from '../src/core/draft.js';
import { requestStance } from '../src/core/stance.js';
import { hitEnemy } from '../src/core/damage.js';

const dt = TICK_DT;

function mkRun(seed, stageId, pos = 0, hooks = { enemies: null, emitters: null, run: tickRun, boss: bossHook }) {
  const w = createWorld({ data: loadData(), seed, weapons, hooks });
  initRun(w);
  if (stageId) w.run.order[pos] = stageId;
  w.run.stageIndex = pos;
  w.player.hp = 100; w.player.hpMax = 100;
  return w;
}
function tick(w, n, input = makeInput()) { for (let i = 0; i < n; i += 1) step(w, input, dt); }
function core(w) { for (const e of w.enemies.items) if (e.alive && e.isBoss && e.isCore) return e; return null; }
function traitPickups(w) { return w.pickups.items.filter((q) => q.alive && q.kind === 'trait'); }

suite('traits — 구슬 (§11.6)', () => {
  test('스테이지 보스 코어 격파 → 금색 구슬이 자석으로 날아와 먹히면 traitQueue 가 1 오르고, 그때까지 STAGE_CLEAR 를 기다린다', () => {
    const w = mkRun(3, 'sea', 0);
    w.run.phase = PHASE.BOSS; w.run.bossSpawned = false; w.run.bossTimer = w.data.stages.phase.bossTimerSec;
    tick(w, 2);
    const c = core(w);
    assert.ne(c, null, '보스가 섰다');
    w.player.x = c.x; w.player.y = c.y + 300;               // 구슬이 «날아올» 거리
    killEnemy(w, c);
    assert.eq(traitPickups(w).length, 1, '구슬 하나');
    assert.eq(traitPickups(w)[0].magnet, true, '자석이 켜져 있다');
    assert.eq(traitPickupAlive(w), true);
    tick(w, 1);
    assert.eq(w.run.phase, PHASE.BOSS, '구슬이 살아 있는 동안은 아직 BOSS (STAGE_CLEAR 대기)');
    let n = 0; while (traitPickupAlive(w) && n < 600) { tick(w, 1); n += 1; }
    assert.ok(n < 600, `구슬이 날아와 먹힌다 (${(n * dt).toFixed(2)}초)`);
    assert.eq(w.traitQueue, 1, '먹으면 traitQueue 1');
    tick(w, 2);
    assert.eq(w.run.phase, PHASE.STAGE_CLEAR, '그다음 STAGE_CLEAR');
  });

  test('최종(테마 없음)에서는 구슬이 없다', () => {
    const w = mkRun(3, 'finale', 5);
    w.run.phase = PHASE.BOSS; w.run.bossSpawned = false; w.run.bossTimer = w.data.stages.phase.bossTimerSec;
    tick(w, 2);
    const c = core(w);
    assert.ne(c, null, '최종 보스');
    killEnemy(w, c);
    assert.eq(traitPickups(w).length, 0, '최종은 구슬 0');
  });
});

suite('traits — 드래프트 (§11.6)', () => {
  test('offerCount 장 · 테마별 하나 우선 · 가진 특성만 제외(같은 테마 여러 개 가능) · applyCard 가 traitQueue 를 소비', () => {
    const w = mkRun(5, 'sea', 0);
    const td = w.data.traits;
    w.traitQueue = 1;
    const d1 = buildTraitDraft(w);
    assert.eq(d1.cards.length, td.offerCount, `${td.offerCount}장`);
    const groups = new Set(d1.cards.map((c) => c.group));
    assert.eq(groups.size, d1.cards.length, '테마가 넉넉하면 한 제안 안의 테마는 전부 다르다');
    for (const c of d1.cards) assert.eq(c.category, 'trait', 'category trait');
    const pick = d1.cards.find((c) => c.group === 'heal') || d1.cards[0];
    applyCard(w, pick);
    assert.eq(w.traitQueue, 0, '큐 소비');
    assert.deepEq(w.traits, [pick.traitId], '보유');
    // 가진 특성은 다시 안 나오고, 같은 테마는 «나온다»(㉑ 묶음 배타 없음)
    let sameGroupSeen = false;
    for (let k = 0; k < 30; k += 1) {
      const d2 = buildTraitDraft(w);
      assert.eq(d2.cards.length, td.offerCount, '항상 offerCount 장');
      for (const c of d2.cards) { assert.ne(c.traitId, pick.traitId, '가진 특성 제외'); if (c.group === pick.group) sameGroupSeen = true; }
    }
    assert.ok(sameGroupSeen, '같은 테마의 다른 특성이 제안된다');
    assert.eq(applyTrait(w, pick.traitId), false, '중복 획득 = false');
    // 같은 테마를 셋 다 가질 수 있다
    const heals = td.traits.filter((t) => t.group === 'heal').map((t) => t.id);
    for (const id of heals) if (id !== pick.traitId) assert.ok(applyTrait(w, id), `${id} 획득`);
    assert.eq(w.traits.length, heals.length + (heals.includes(pick.traitId) ? 0 : 1), '회복 3종 전부 보유');
  });

  test('테마가 바닥나도 제안은 offerCount 장 — 5번째 보스까지 3택 (S59 ③ 의 런타임 대응)', () => {
    const w = mkRun(6, 'sea', 0);
    const td = w.data.traits;
    // 두 테마를 통째로 비운다(6개 보유 — 실제 런의 최대 5개보다 가혹한 조건)
    const gs = td.groups.slice(0, 2);
    for (const t of td.traits) if (gs.includes(t.group)) assert.ok(applyTrait(w, t.id));
    const d = buildTraitDraft(w);
    assert.eq(d.cards.length, td.offerCount, `남은 테마 ${td.groups.length - 2}개여도 ${td.offerCount}장`);
    for (const c of d.cards) assert.ok(!gs.includes(c.group), '비운 테마는 안 나온다(가진 특성뿐이라)');
    // 실제 런: 5개를 임의로 가진 뒤에도 3장
    const w2 = mkRun(7, 'sea', 0);
    for (let k = 0; k < 5; k += 1) { const dk = buildTraitDraft(w2); assert.eq(dk.cards.length, td.offerCount, `${k + 1}번째 보스: 3장`); applyTrait(w2, dk.cards[0].traitId); }
    assert.eq(w2.traits.length, 5, '5개 보유');
  });

  test('결정성 — 같은 시드 = 같은 제안', () => {
    const a = buildTraitDraft(mkRun(9, 'sea', 0)).cards.map((c) => c.traitId).join(',');
    const b = buildTraitDraft(mkRun(9, 'sea', 0)).cards.map((c) => c.traitId).join(',');
    assert.eq(a, b);
  });
});

suite('traits — 효과 (§11.6)', () => {
  test('자연 재생 — 초당 value 만큼, hpMax 에서 멈춘다', () => {
    const w = mkRun(1, 'sea', 0);
    const def = w.data.traits.traits.find((t) => t.id === 'regen');
    applyTrait(w, 'regen');
    w.player.hp = 50;
    tick(w, 60);
    assert.near(w.player.hp, 50 + def.effect.value, 0.05, '1초에 value');
    w.player.hp = 99.9; tick(w, 60);
    assert.eq(w.player.hp, 100, '상한');
  });

  test('회수 — 잡몹 N 마리마다 HP 1 (유령·중간보스 제외)', () => {
    const w = mkRun(1, 'sea', 0);
    const N = w.data.traits.traits.find((t) => t.id === 'salvage').effect.value;
    applyTrait(w, 'salvage');
    w.player.hp = 50;
    const dd = w.data.enemies.archetypes.find((a) => a.id === 'drifter');
    for (let i = 0; i < N - 1; i += 1) killEnemy(w, spawnEnemy(w, 'drifter', 'water', 640, 300, dd.hp, false));
    assert.eq(w.player.hp, 50, `${N - 1}마리까진 없음`);
    killEnemy(w, spawnEnemy(w, 'drifter', 'water', 640, 300, dd.hp, false));
    assert.eq(w.player.hp, 51, `${N}마리째 +1`);
    const g = spawnEnemy(w, 'drifter', 'water', 640, 300, dd.hp, false, true);   // 유령
    for (let i = 0; i < N; i += 1) { const e = spawnEnemy(w, 'drifter', 'water', 640, 300, dd.hp, false, true); killEnemy(w, e); }
    void g;
    assert.eq(w.player.hp, 51, '유령은 안 센다');
  });

  test('보급 강화 — 스테이지 클리어 회복 비율이 특성 값으로', () => {
    const w = mkRun(1, 'sea', 0);
    const v = w.data.traits.traits.find((t) => t.id === 'resupply').effect.value;
    w.player.hp = 10; applyStageClearHeal(w);
    assert.near(w.player.hp, 10 + w.data.meta.flow.stageClearHealPct * 100, 1e-9, '기본');
    applyTrait(w, 'resupply');
    w.player.hp = 10; applyStageClearHeal(w);
    assert.near(w.player.hp, 10 + v * 100, 1e-9, '특성 값');
  });

  test('격벽 — everySec 뒤 방패 1개, 피격 1회를 통째로 막고 다시 충전한다', () => {
    const w = mkRun(1, 'sea', 0);
    const every = w.data.traits.traits.find((t) => t.id === 'barrier').effect.value;
    applyTrait(w, 'barrier');
    assert.eq(w.traitState.barrierReady, false, '처음엔 없다');
    tick(w, Math.round(every / dt) + 2);
    assert.eq(w.traitState.barrierReady, true, 'everySec 뒤 충전');
    w.player.iframeSec = 0;
    assert.eq(applyHit(w, 30, ''), true, '피격 처리됨');
    assert.eq(w.player.hp, 100, '피해 0');
    assert.eq(w.traitState.barrierReady, false, '소모');
    assert.gt(w.player.iframeSec, 0, 'i-frame 은 그대로(연타 차단)');
    w.player.iframeSec = 0;
    applyHit(w, 30, '');
    assert.lt(w.player.hp, 100, '방패 없으면 맞는다');
  });

  test('재기 — 치명상을 HP 1 로 버티고 긴 무적, 스테이지당 한 번, advanceStage 가 리셋', () => {
    const w = mkRun(1, 'sea', 0);
    const ifr = w.data.traits.traits.find((t) => t.id === 'secondWind').effect.value;
    applyTrait(w, 'secondWind');
    w.player.hp = 5; w.player.iframeSec = 0;
    applyHit(w, 999, '');
    assert.eq(w.player.hp, 1, 'HP 1');
    assert.eq(w.over, false, '안 죽었다');
    assert.near(w.player.iframeSec, ifr, 1e-9, '무적');
    w.player.iframeSec = 0;
    applyHit(w, 999, '');
    assert.eq(w.over, true, '두 번째는 죽는다');
    // 리셋
    const w2 = mkRun(1, 'sea', 0); applyTrait(w2, 'secondWind'); w2.player.hp = 5; w2.player.iframeSec = 0; applyHit(w2, 999, '');
    assert.eq(w2.traitState.secondWindUsed, true); advanceStage(w2);
    assert.eq(w2.traitState.secondWindUsed, false, 'advanceStage 리셋');
  });

  test('청정 · 보스 사냥꾼 — 피해 입구(hitEnemy)에서만 배율', () => {
    const w = mkRun(1, 'sea', 0);
    const dd = w.data.enemies.archetypes.find((a) => a.id === 'drifter');
    const e = spawnEnemy(w, 'drifter', 'normal', 640, 300, 1e6, false);
    const ctx = w.dmgCtx; ctx.matrix = w.data.elements.matrix;
    const base = hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0);
    applyTrait(w, 'clean');
    const v = w.data.traits.traits.find((t) => t.id === 'clean').effect;
    w.player.hp = 100;
    const up = hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0);
    assert.near(up, base * (1 + v.value), 1e-6, 'HP 90%+ → +15%');
    w.player.hp = 50;
    assert.near(hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0), base, 1e-6, 'HP 낮으면 없음');
    void dd;
    const w2 = mkRun(2, 'sea', 0); const ctx2 = w2.dmgCtx; ctx2.matrix = w2.data.elements.matrix;
    const bv = w2.data.traits.traits.find((t) => t.id === 'bossHunter').effect.value;
    w2.run.phase = PHASE.BOSS; w2.run.bossSpawned = false; w2.run.bossTimer = w2.data.stages.phase.bossTimerSec; tick(w2, 2);
    const c = core(w2); w2.run.bossTransitionT = 0; c.hp = 1e6;
    const b0 = hitEnemy(w2, ctx2, 'forward', 10, 1, 'normal', c, 0);
    applyTrait(w2, 'bossHunter');
    const b1 = hitEnemy(w2, ctx2, 'forward', 10, 1, 'normal', c, 0);
    assert.gt(b0, 0, '보스에 피해가 든다');
    assert.near(b1, b0 * (1 + bv), 1e-6, '같은 보스에 +25%');
    const mob = spawnEnemy(w2, 'drifter', 'normal', 640, 300, 1e6, false);
    const m1 = hitEnemy(w2, ctx2, 'forward', 10, 1, 'normal', mob, 0);
    void m1;
  });

  test('스탠스 공명 — 전환 순간 반경 안의 적 탄 소거 + 무적', () => {
    const w = mkRun(1, 'sea', 0);
    const ef = w.data.traits.traits.find((t) => t.id === 'stanceEcho').effect;
    applyTrait(w, 'stanceEcho');
    const p = w.player;
    spawnEnemyBullet(w, 'pelletS', p.x + 40, p.y, 0, 0);
    spawnEnemyBullet(w, 'pelletS', p.x + ef.value + 60, p.y, 0, 0);
    p.iframeSec = 0;
    assert.ok(requestStance(w, 'fire'), '전환');
    assert.eq(w.enemyBullets.live, 1, '반경 안 1발만 지워졌다');
    assert.near(p.iframeSec, ef.iframeSec, 1e-9, '무적');
  });

  test('장갑 — 정액 감산 + 원본의 damageFloorRatio 하한 · 획득이 player.defense 의 단일 소유자', () => {
    const w = mkRun(1, 'sea', 0);
    const rp = w.data.rules.player;
    const v = w.data.traits.traits.find((t) => t.id === 'armor').effect.value;
    assert.eq(w.player.defense, rp.defenseBase, '기본 방어');
    w.player.iframeSec = 0; applyHit(w, 10, '');
    assert.eq(w.player.hp, 90, '기본: 10 그대로');
    applyTrait(w, 'armor');
    assert.eq(w.player.defense, rp.defenseBase + v, `방어 +${v}`);
    w.player.hp = 100; w.player.iframeSec = 0; applyHit(w, 10, '');
    assert.eq(w.player.hp, 100 - (10 - v), `10 → ${10 - v}`);
    w.player.hp = 100; w.player.iframeSec = 0; applyHit(w, 2, '');
    assert.eq(w.player.hp, 100 - Math.ceil(2 * rp.damageFloorRatio), `작은 탄도 하한(${rp.damageFloorRatio}) 아래로는 안 내려간다`);
  });

  test('처형 — 잔여 HP 비율 ≤ hpRatio 인 «그 개체»에만 +N%', () => {
    const w = mkRun(1, 'sea', 0);
    const ef = w.data.traits.traits.find((t) => t.id === 'execute').effect;
    const ctx = w.dmgCtx; ctx.matrix = w.data.elements.matrix;
    const e = spawnEnemy(w, 'drifter', 'normal', 640, 300, 1000, false);
    const base = hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0);
    applyTrait(w, 'execute');
    e.hp = e.hpMax * (ef.hpRatio + 0.2);
    assert.near(hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0), base, 1e-6, '아직 높으면 없음');
    e.hp = e.hpMax * ef.hpRatio;
    assert.near(hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e, 0), base * (1 + ef.value), 1e-6, `≤ ${ef.hpRatio} → +${ef.value * 100}%`);
    const e2 = spawnEnemy(w, 'drifter', 'normal', 600, 300, 1000, false);
    assert.near(hitEnemy(w, ctx, 'forward', 10, 1, 'normal', e2, 0), base, 1e-6, '다른(멀쩡한) 개체는 그대로');
  });

  test('전환 가속 — 전환 뒤 sec 동안 발사 주기가 (1 + fireRateMul + value) 로 나뉜다 · 창이 지나면 원래대로', () => {
    const w = mkRun(1, 'sea', 0);
    const ef = w.data.traits.traits.find((t) => t.id === 'stanceSurge').effect;
    const slot = w.slots.find((s2) => s2.weaponId !== null);
    assert.ok(slot !== undefined, '시작 무기가 있다');
    const hooks = w.data.rules.passiveHooks[slot.family];
    const r0 = recomputeEff(w, slot)[hooks.rateKey];
    applyTrait(w, 'stanceSurge');
    assert.near(recomputeEff(w, slot)[hooks.rateKey], r0, 1e-9, '전환 전엔 그대로');
    assert.ok(requestStance(w, 'fire'), '전환');
    assert.near(w.traitState.surgeT, ef.sec, 1e-9, '창 시작');
    assert.near(recomputeEff(w, slot)[hooks.rateKey], r0 / (1 + ef.value), 1e-9, `주기 ÷ (1 + ${ef.value})`);
    tick(w, Math.round(ef.sec / dt) + 2);
    assert.eq(w.traitState.surgeT, 0, '창 종료');
    assert.near(recomputeEff(w, slot)[hooks.rateKey], r0, 1e-9, '원래대로');
  });

  test('전환 회수 — 전환 순간 반경 안의 픽업만 자석에 붙는다', () => {
    const w = mkRun(1, 'sea', 0);
    const r = w.data.traits.traits.find((t) => t.id === 'stanceMagnet').effect.value;
    const p = w.player;
    const near = spawnPickup(w, 'xp', 1, p.x + r - 20, p.y - 100);
    const far = spawnPickup(w, 'xp', 1, p.x, p.y - r - 60);
    assert.eq(near.magnet, false); assert.eq(far.magnet, false);
    requestStance(w, 'water');
    assert.eq(near.magnet, false, '특성 없으면 안 붙는다');
    applyTrait(w, 'stanceMagnet');
    requestStance(w, 'fire');
    assert.eq(near.magnet, true, '반경 안 = 자석');
    assert.eq(far.magnet, false, '반경 밖 = 그대로');
  });
});

