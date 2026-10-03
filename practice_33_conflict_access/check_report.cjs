const {chromium}=require(process.env.P33_BROWSER_PACKAGE||'/tmp/inference-report-check/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const report=path.resolve(process.argv[2]);
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1400,height:1000},offline:true});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()));
  await page.goto('file://'+report);
  await page.waitForSelector('rect.task');
  const cases=await page.locator('#case option').count();
  if(cases!==3)throw Error('Expected three profile cases');
  for(let i=0;i<cases;i++){
   await page.selectOption('#case',String(i));
   if((await page.locator('#trial option').count())!==2)throw Error('Expected two trials per case');
   for(const trial of ['0','1'])for(const window of ['full','release','conflict']){
    await page.selectOption('#trial',trial);
    await page.selectOption('#window',window);
    if(!(await page.locator('rect.task').count()))throw Error('Empty timeline window');
    await page.locator('rect.task').first().click();
    if(!(await page.locator('#details').textContent()).includes('task_id')&&!(await page.locator('#details').textContent()).includes('pointer'))throw Error('Missing evidence');
   }
  }
  await page.selectOption('#case','0');await page.selectOption('#window','full');
  const status=await page.locator('#status').textContent();
  if(!status.includes('observed'))throw Error('Missing outcome in status line');
  await page.screenshot({path:'/tmp/p33-conflict-timeline.png',fullPage:false});
  await page.setViewportSize({width:780,height:920});
  if(errors.length)throw Error(errors.join('\n'));
  const result={report,offline:true,cases,all_case_trial_window_controls:true,click_evidence:true,errors};
  console.log(JSON.stringify(result,null,2));
  if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2)+'\n');
 }finally{await browser.close()};
})().catch(e=>{console.error(e);process.exit(1)});
