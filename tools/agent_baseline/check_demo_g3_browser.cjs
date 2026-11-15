/* Run on a compute node with Playwright and Chromium available. */
const assert = require('node:assert/strict');
const { writeFile } = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const [site, report] = process.argv.slice(2);
if (!site || !report) {
  throw new Error('usage: node check_demo_g3_browser.cjs <demo_g3 directory> <report.json>');
}
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const entry = pathToFileURL(path.resolve(site, 'index.html')).href;

async function check() {
  const browser = await chromium.launch({ headless: true });
  try {
    const screenshotDirectory = path.dirname(path.resolve(report));
    const screenshot = (page, name) => page.screenshot({
      path: path.join(screenshotDirectory, name), animations: 'disabled',
    });
    const screenshots = ['catalog-desktop.png', 'catalog-replay.png', 'catalog-review.png', 'catalog-mobile.png'];
    const page = await browser.newPage({ offline: true, viewport: { width: 1360, height: 1000 } });
    const errors = [];
    const remote = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (/^https?:/.test(request.url())) remote.push(request.url());
    });
    await page.goto(entry);
    const rows = page.locator('[data-replay-row]');
    const visible = page.locator('[data-replay-row]:visible');
    const total = await rows.count();
    assert(total > 9);
    await screenshot(page, screenshots[0]);

    const filter = name => page.locator(`#catalog-filters select[name="${name}"]`);
    await filter('task').selectOption('T1.2');
    await filter('constraint').selectOption('C0');
    await filter('dataset').selectOption('SECOND');
    await filter('harness').selectOption('G3');
    assert((await visible.count()) >= 2);
    for (const row of await visible.all()) {
      assert.equal(await row.getAttribute('data-task'), 'T1.2');
      assert.equal(await row.getAttribute('data-constraint'), 'C0');
      assert.equal(await row.getAttribute('data-dataset'), 'SECOND');
      assert.equal(await row.getAttribute('data-harness'), 'G3');
    }
    await filter('constraint').selectOption('C3');
    assert.equal(await visible.count(), 0);
    assert(await page.locator('#filter-empty').isVisible());
    await page.getByRole('button', { name: '清除筛选' }).click();
    await page.waitForFunction(count => document.querySelectorAll('[data-replay-row]:not([hidden])').length === count, total);
    await page.locator('[data-coverage-task="T1.2"][data-coverage-constraint="C0"]').click();
    assert.equal(await filter('task').inputValue(), 'T1.2');
    assert.equal(await filter('constraint').inputValue(), 'C0');

    await visible.first().getByRole('link').click();
    const firstReplay = page.url();
    const replayImage = page.locator('.inputs button.zoom').first();
    await replayImage.locator('img').evaluate(image => image.decode());
    await replayImage.click();
    assert(await page.locator('#zoom-dialog').isVisible());
    await page.keyboard.press('Escape');
    assert(!(await page.locator('#zoom-dialog').isVisible()));
    const step = page.locator('[data-step="0"] [data-select]').first();
    await step.click();
    assert.match(await page.locator('#current-label').textContent(), /第 0 步/);
    await page.keyboard.press('ArrowRight');
    assert.match(await page.locator('#current-label').textContent(), /第 1 步/);
    const sibling = page.locator('.replay-switcher a').first();
    await sibling.click();
    assert.notEqual(page.url(), firstReplay);
    assert(await page.locator('.replay-switcher [aria-current="page"]').isVisible());
    await page.locator('main.layout').evaluate(element => element.scrollIntoView());
    await screenshot(page, screenshots[1]);

    await page.goto(new URL('data_review.html', entry).href);
    assert(await page.locator('[id="T4.3"]').isVisible());
    await page.locator('.review-case img').first().evaluate(image => image.decode());
    await page.locator('.review-case button.zoom').first().click();
    assert(await page.locator('#review-zoom').isVisible());
    await page.keyboard.press('Escape');
    await screenshot(page, screenshots[2]);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(entry);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
    assert.equal(await visible.count(), total);
    for (const section of ['#replays', '#coverage']) {
      const table = page.locator(`${section} .tablewrap`);
      const layout = await table.evaluate(element => {
        const initialLeft = element.scrollLeft;
        element.scrollLeft = 200;
        const result = {
          client: element.clientWidth, width: element.scrollWidth,
          minColumn: Math.min(...[...element.querySelectorAll('thead th')].map(cell => cell.getBoundingClientRect().width)),
          scrolled: element.scrollLeft > initialLeft,
        };
        element.scrollLeft = initialLeft;
        return result;
      });
      assert(layout.width > layout.client && layout.scrolled, `Horizontal scrolling is missing: ${section}`);
      assert(layout.minColumn >= 72, `Columns are too narrow: ${section}`);
    }
    await screenshot(page, screenshots[3]);
    assert.deepEqual(errors, []);
    assert.deepEqual(remote, []);
    const result = { status: 'passed', core_replays: total, browser: browser.version(), offline: true, screenshots,
      checks: ['four filters', 'empty state', 'reset', 'coverage shortcut', 'same-group switch', 'replay step and keyboard',
        'replay image and zoom', 'review image and zoom', 'mobile page width and scrollable tables', 'no remote resource requests'] };
    await writeFile(report, JSON.stringify(result, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}

check().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
