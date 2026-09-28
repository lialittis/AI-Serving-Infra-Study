const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');
const assert=require('assert'), fs=require('fs');
const root='/home/tianchi/workspace/AI-Serving-Infra-Study/practice_15_kernel_execution_graph';
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});let errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file://'+root+'/results/2026-09-28-mode-comparison/index.html');
 for(const link of await page.locator('a').evaluateAll(xs=>xs.map(x=>x.href)))assert(fs.existsSync(new URL(link)));
 await page.screenshot({path:'/tmp/p15-modes.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 for(const mode of ['eager','graph']){
  await page.setViewportSize({width:1440,height:1100});
  await page.goto('file://'+root+'/results/2026-09-28-'+mode+'-run01/analysis/index.html');
  assert.equal(await page.locator('.task').count(),24);assert.equal(await page.locator('#phase option').count(),5);
  await page.locator('#phase').selectOption('decode-1');
  await page.locator('#search').fill(mode==='graph'?'MODEL_EXECUTE':'ReshapeAndCache');
  await page.locator('.task').first().focus();await page.keyboard.press('Enter');
  const detail=JSON.parse(await page.locator('#detail').textContent());
  assert.equal(detail.node.phase,'decode-1');
  if(mode==='graph'){
   assert(detail.node.runtime_connection);assert.equal(detail.submission.name,'AscendCL@aclmdlRIExecuteAsync');
   assert.equal(await page.locator('#replays tr').count(),75);
   await page.locator('#replays tr').first().click();const replay=JSON.parse(await page.locator('#detail').textContent());assert(replay.replay.resources_verified);assert.equal(replay.boundary_tasks.length,2);
   await page.locator('#search').fill('_triton_rope_1');await page.locator('.task').first().click();const unknown=JSON.parse(await page.locator('#detail').textContent());assert(!unknown.host);assert(!unknown.submission);
  }
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 }
 assert.deepEqual(errors,[]);assert(requests.every(x=>x.startsWith('file:')));
 console.log(JSON.stringify({offline:true,mode_pages:2,phase_filters:5,replays:75,exact_runtime_chain:true,unknown_not_fabricated:true,keyboard:true,mobile_no_overflow:true,errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
