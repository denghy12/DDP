const { chromium } = require('playwright');
const path = require('path');

async function main() {
  const root = path.resolve(__dirname);
  const htmlPath = path.join(root, 'emotic_full_base5_result_tables.html');
  const outputs = [
    ['#full-base5-table', 'code_ddp_full_base5_cls_correction_results_table.png'],
    ['#ablation-table', 'code_ddp_internal_adapter_transfer_ablation_table_updated.png'],
  ];
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  const page = await browser.newPage({
    viewport: { width: 2100, height: 1700 },
    deviceScaleFactor: 2,
  });
  await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  for (const [selector, filename] of outputs) {
    const element = page.locator(selector);
    await element.screenshot({ path: path.join(root, filename) });
  }
  await browser.close();
  process.stdout.write(outputs.map(([, file]) => path.join(root, file)).join('\n') + '\n');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
