const { chromium } = require('playwright');
const path = require('path');

const chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const root = path.resolve(__dirname);
const htmlPath = path.join(
  root,
  'emotic_cls_weights_to_final_tokens_architecture.html',
);

const variants = [
  {
    mode: 'all_tokens',
    stem: 'code_ddp_emotic_cls_weights_all_final_tokens_architecture',
  },
  {
    mode: 'cls_only',
    stem: 'code_ddp_emotic_cls_weights_cls_token_only_architecture',
  },
  {
    mode: 'pooling_aware',
    stem: 'code_ddp_emotic_final_token_pooling_aware_prompted_197_tokens_architecture',
  },
];

async function main() {
  const requestedMode = process.env.RENDER_MODE;
  const selected = requestedMode
    ? variants.filter((variant) => variant.mode === requestedMode)
    : variants;
  if (!selected.length) {
    throw new Error(`Unknown RENDER_MODE: ${requestedMode}`);
  }
  for (const variant of selected) {
    process.stdout.write(`Rendering ${variant.mode}: launch\n`);
    const browser = await chromium.launch({
      headless: true,
      executablePath: chrome,
    });
    const page = await browser.newPage({
      viewport: { width: 3800, height: 2850 },
      deviceScaleFactor: 2,
    });
    page.on('pageerror', (error) => {
      process.stderr.write(`Page error (${variant.mode}): ${error.stack}\n`);
    });
    page.on('console', (message) => {
      if (message.type() === 'error') {
        process.stderr.write(
          `Browser console (${variant.mode}): ${message.text()}\n`,
        );
      }
    });
    page.setDefaultNavigationTimeout(30000);
    process.stdout.write(`Rendering ${variant.mode}: load HTML\n`);
    await page.goto(
      `file://${htmlPath}?mode=${variant.mode}`,
      { waitUntil: 'load' },
    );
    process.stdout.write(`Rendering ${variant.mode}: wait for formulas\n`);
    await page.waitForFunction(
      () => window.__diagramReady === true,
      null,
      { timeout: 30000 },
    );
    process.stdout.write(`Rendering ${variant.mode}: PNG\n`);
    const pngPath = path.join(root, `${variant.stem}.png`);
    const pdfPath = path.join(root, `${variant.stem}.pdf`);
    await page.screenshot({ path: pngPath, fullPage: true });
    process.stdout.write(`Rendering ${variant.mode}: PDF\n`);
    await page.pdf({
      path: pdfPath,
      width: '3800px',
      height: '2850px',
      printBackground: true,
      margin: { top: '0px', right: '0px', bottom: '0px', left: '0px' },
    });
    process.stdout.write(`${pngPath}\n${pdfPath}\n`);
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
