// Offline browser check. P22_BROWSER_PACKAGE can point to another Playwright installation.
const {chromium}=require(process.env.P22_BROWSER_PACKAGE||'/tmp/inference-report-check/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()+': '+r.failure().errorText));
 await page.goto('file://'+path.join(__dirname,'report/index.html'));
 let tasks=0;
 for(const c of ['eager','graph','sampling-off','sampling-on']){
  await page.locator('#case').selectOption(c);
  if(!(await page.locator('#inventory tr').count()))throw Error('missing streams '+c);
  await page.locator('[data-origin]').first().click();
  if(!(await page.locator('#origin').textContent()).includes('stream'))throw Error('missing source');
  await page.locator('#stream').selectOption('all');await page.locator('#phase').selectOption('all');
  await page.locator('[data-task]').first().click();
  if(!(await page.locator('#taskDetail').textContent()).includes('trace_index'))throw Error('missing task evidence');
  tasks+=await page.locator('[data-task]').count();
 }
 await page.locator('#case').selectOption('graph');
 await page.locator('#stream').selectOption('10');
 await page.screenshot({path:'/tmp/p22-report.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
 await page.goto('file://'+path.join(__dirname,'report/graph_replay_sequence.svg'));
 if(!(await page.locator('svg').textContent()).includes('submod_0'))throw Error('missing graph sequence');
 if(errors.length)throw Error(errors.join('\n'));
 const result={offline:true,cases:4,task_buttons_checked:tasks,stream_sources:true,mobile_overflow:false,svg:true,page_errors:errors};
 const out=path.join(__dirname,'results/validation');fs.mkdirSync(out,{recursive:true});
 fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(result,null,2)+'\n');console.log(result);
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
