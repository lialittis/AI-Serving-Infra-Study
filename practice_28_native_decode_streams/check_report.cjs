const {chromium} = require(process.env.P28_BROWSER_PACKAGE || '/tmp/inference-report-check/node_modules/playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const browser = await chromium.launch({headless:true});
  const page = await browser.newPage({viewport:{width:1440,height:1000},offline:true});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('requestfailed', r => errors.push(r.url()));
  await page.goto('file://' + path.join(__dirname,'report/index.html'));
  const figures = await page.locator('svg').count();
  if (figures !== 32) throw Error('Expected 32 independent diagnostic figures');
  for (const mode of ['eager','graph']) {
    const detail = page.locator('details').filter({has:page.locator('summary', {hasText:mode+'-plain-same32-parallel-AB · overlap'})}).first();
    await detail.locator('summary').click();
    await detail.scrollIntoViewIfNeeded();
    if (await detail.locator('rect title').count() !== 585) throw Error('Missing 576 compute tasks or 9 CPU scopes');
    await page.screenshot({path:'/tmp/p28-'+mode+'-report.png'});
    await detail.locator('summary').click();
  }
  await page.goto('file://' + path.join(__dirname,'report/graph-plain-same32-parallel-AB.svg'));
  if (!(await page.locator('svg').textContent()).includes('CPU submission')) throw Error('SVG CPU lane missing');
  if (errors.length) throw Error(errors.join('\n'));
  const result = {offline:true, figures, raw_task_tooltips:true, standalone_svg:true, errors};
  const out = path.join(__dirname,'results/validation');
  fs.mkdirSync(out,{recursive:true});
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(result,null,2)+'\n');
  console.log(result);
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
