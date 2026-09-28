import unittest

from analyze_run import placeholder_task, resolve_colliding_host


class FlowCollisionTests(unittest.TestCase):
    def setUp(self):
        self.hosts = [(1, {"pid": 10, "tid": 10, "ts": 100, "dur": 2}),
                      (2, {"pid": 10, "tid": 10, "ts": 110, "dur": 2})]
        self.links = [({"pid": 10, "tid": 11, "ts": 200, "dur": 8},
                       {"pid": 10, "tid": 10, "ts": 101}, "a"),
                      ({"pid": 10, "tid": 11, "ts": 210, "dur": 8},
                       {"pid": 10, "tid": 10, "ts": 111}, "b")]

    def test_queue_evidence_selects_earlier_host_not_nearest(self):
        cann = {"pid": 999, "tid": 11, "ts": 201, "dur": 2}
        result = resolve_colliding_host(self.hosts, cann, self.links)
        self.assertEqual((result[0], result[2]), (1, "a"))

    def test_missing_or_ambiguous_evidence_fails(self):
        cann = {"pid": 999, "tid": 11, "ts": 201, "dur": 2}
        with self.assertRaises(ValueError): resolve_colliding_host(self.hosts, cann, [])
        with self.assertRaises(ValueError): resolve_colliding_host(self.hosts, cann, self.links * 2)

    def test_wrong_thread_or_out_of_scope_not_accepted(self):
        for cann in [{"pid": 999, "tid": 12, "ts": 201, "dur": 2},
                     {"pid": 999, "tid": 11, "ts": 201, "dur": 20}]:
            with self.assertRaises(ValueError): resolve_colliding_host(self.hosts, cann, self.links)

    def test_placeholder_has_no_invented_kernel_or_host_ownership(self):
        task = {"name": "PLACE_HOLDER_SQE", "pid": 999, "tid": 46, "ts": "120.123", "dur": .02,
                "args": {"Task Type": "PLACE_HOLDER_SQE", "Task Id": 0,
                         "Physic Stream Id": 46, "connection_id": 2**64-1}}
        node = placeholder_task(7, task)
        self.assertEqual(node["end_us"], "120.143")
        self.assertFalse(node["is_compute"])
        self.assertIsNone(node["step"])
        self.assertIsNone(node["torch_flow"])
        self.assertEqual(node["stream"], "46")
        task["args"]["connection_id"] = 123
        with self.assertRaises(ValueError): placeholder_task(7, task)


if __name__ == "__main__":
    unittest.main()
