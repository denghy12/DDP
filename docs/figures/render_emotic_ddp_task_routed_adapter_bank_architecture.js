const { chromium } = require('playwright');
const path = require('path');

async function main() {
  const root = path.resolve(__dirname);
  const htmlPath = path.join(root, 'emotic_ddp_task_routed_adapter_bank_architecture.html');
  const pngPath = path.join(root, 'code_ddp_emotic_task_routed_adapter_bank_architecture.png');
  const pdfPath = path.join(root, 'code_ddp_emotic_task_routed_adapter_bank_architecture.pdf');

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  const page = await browser.newPage({
    viewport: { width: 3200, height: 2300 },
    deviceScaleFactor: 2,
  });
  await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__diagramReady === true, null, { timeout: 30000 });
  await page.screenshot({ path: pngPath, fullPage: true });
  await page.pdf({
    path: pdfPath,
    width: '3200px',
    height: '2300px',
    printBackground: true,
    margin: { top: '0px', right: '0px', bottom: '0px', left: '0px' },
  });
  await browser.close();
  process.stdout.write(`${pngPath}\n${pdfPath}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
