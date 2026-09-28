// Optional offline browser verification; set P20_BROWSER_PACKAGE if not installed locally.
const {chromium}=require(process.env.P20_BROWSER_PACKAGE || 'playwright');
const assert=require('assert'),path=require('path'),{pathToFileURL}=require('url');
(async()=>{
 const browser=await chromium.launch({headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1400,height:1000},offline:true});
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.goto(pathToFileURL(path.resolve(__dirname,'report/index.html')).href);
  assert.equal(await page.locator('#run option').count(),3);
  let transfers=0,blocks=0;
  for(let run=0;run<3;run++){
   await page.locator('#run').selectOption(String(run));
   const choices=await page.locator('#transfer option').evaluateAll(xs=>xs.map(x=>x.value));
   assert.equal(choices.length,run===2?0:18);
   await page.locator('#zoom').selectOption('100');
   for(const t of choices){
    await page.locator('#transfer').selectOption(t);
    const bars=page.locator('[data-task]');assert(await bars.count()>0);
    const index=await bars.evaluateAll(xs=>xs.findIndex(e=>Number(e.getAttribute('width'))>1));
    assert(index>=0);await bars.nth(index).click();
    assert(JSON.parse(await page.locator('#detail').textContent()).host_anchor_kind);
    transfers++;
   }
   for(const medium of ['NPU','CPU']){
    await page.locator('#medium').selectOption(medium);
    const values=await page.locator('#block option').evaluateAll(xs=>xs.map(x=>x.value));
    for(const value of values.slice(0,3)){
     await page.locator('#block').selectOption(value);
     const nodes=page.locator('[data-node]');assert(await nodes.count()>0);
     await nodes.first().click();assert(JSON.parse(await page.locator('#detail').textContent()).dependencies.length);
     blocks++;
    }
   }
  }
  await page.locator('#run').selectOption('0');await page.locator('#transfer').selectOption('7');
  await page.locator('#zoom').selectOption('1');
  await page.screenshot({path:'/tmp/p20-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.screenshot({path:'/tmp/p20-mobile.png',fullPage:true});
  assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
  console.log(JSON.stringify({offline:true,modes:3,transfer_filters:transfers,block_filters:blocks,
    node_details:true,mobile_no_overflow:true,page_errors:errors}));
 } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
