// #CSR = #sparse_tensor.encoding<{
//   map = (d0, d1) -> (d0 : dense, d1 : compressed)
// }>

// module {
//     func.func @bfs(%nodes_i32: i32, %edges_i32: i32, %coords_mem: memref<?x2xi64>, %vals_mem: memref<?xf32>, %out_dist: memref<?xf32>) attributes { llvm.emit_c_interface } {
        
//         // --- Cast scalar bounds to MLIR indexes ---
//         %num_nodes = arith.index_cast %nodes_i32 : i32 to index
//         %num_edges = arith.index_cast %edges_i32 : i32 to index

//         %c0 = arith.constant 0 : index
//         %c1 = arith.constant 1 : index
        
//         %f0 = arith.constant 0.0 : f32
//         %f1 = arith.constant 1.0 : f32
//         %f_neg1 = arith.constant -1.0 : f32

//         // --- 1. Load C++ Memory into MLIR Tensors ---
//         // Load Coordinates
//         %empty_coords = tensor.empty(%num_edges) : tensor<?x2xi64>
//         %coords_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_coords = %empty_coords) -> tensor<?x2xi64> {
//             %row = memref.load %coords_mem[%e, %c0] : memref<?x2xi64>
//             %col = memref.load %coords_mem[%e, %c1] : memref<?x2xi64>
//             %t1 = tensor.insert %row into %t_coords[%e, %c0] : tensor<?x2xi64>
//             %t2 = tensor.insert %col into %t1[%e, %c1] : tensor<?x2xi64>
//             scf.yield %t2 : tensor<?x2xi64>
//         }

//         // Load Values
//         %empty_vals = tensor.empty(%num_edges) : tensor<?xf32>
//         %vals_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_vals = %empty_vals) -> tensor<?xf32> {
//             %v = memref.load %vals_mem[%e] : memref<?xf32>
//             %t1 = tensor.insert %v into %t_vals[%e] : tensor<?xf32>
//             scf.yield %t1 : tensor<?xf32>
//         }

//         // Generate the #CSR sparse tensor with dynamic dimensions
//         %A = gblas.from_coo %coords_tensor, %vals_tensor 
//             : tensor<?x2xi64>, tensor<?xf32> -> tensor<?x?xf32, #CSR>


//         // --- 2. Initial State Allocation ---
//         %empty_f32 = tensor.empty(%num_nodes) : tensor<?xf32>
        
//         // v_start = [1.0, 0.0, 0.0, ...]
//         %v_start_0 = linalg.fill ins(%f0 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
//         %v_start = tensor.insert %f1 into %v_start_0[%c0] : tensor<?xf32>
        
//         // visited_start = [1.0, 0.0, 0.0, ...]
//         %visited_start_0 = linalg.fill ins(%f0 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
//         %visited_start = tensor.insert %f1 into %visited_start_0[%c0] : tensor<?xf32>
        
//         // dist_start = [0.0, -1.0, -1.0, ...]
//         %dist_start_neg = linalg.fill ins(%f_neg1 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
//         %dist_start = tensor.insert %f0 into %dist_start_neg[%c0] : tensor<?xf32>

//         // --- 3. BFS Distance Loop ---
//         %final_results:3 = scf.for %i = %c0 to %num_nodes step %c1 
//             iter_args(%v_curr = %v_start, %visited_curr = %visited_start, %dist_curr = %dist_start) 
//             -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
            
//             %scratch_empty = tensor.empty(%num_nodes) : tensor<?xf32>
//             %zeroed_scratch = linalg.fill ins(%f0 : f32) outs(%scratch_empty : tensor<?xf32>) -> tensor<?xf32>

//             %v_computed = gblas.vxm %v_curr, %A outs(%zeroed_scratch), %visited_curr
//                 combine = multiplies 
//                 reduce = plus 
//                 {mask_complement = true}
//                 : tensor<?xf32>, tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32> 
//                 -> tensor<?xf32>

//             %i_i32 = arith.index_cast %i : index to i32
//             %i_f32 = arith.sitofp %i_i32 : i32 to f32
//             %d_f32 = arith.addf %i_f32, %f1 : f32

//             %v_next, %visited_next, %dist_next = scf.for %j = %c0 to %num_nodes step %c1 
//                 iter_args(%v_acc = %v_curr, %vis_acc = %visited_curr, %d_acc = %dist_curr) 
//                 -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
                
//                 %val = tensor.extract %v_computed[%j] : tensor<?xf32>
//                 %is_active = arith.cmpf ogt, %val, %f0 : f32
                
//                 %new_dist, %new_vis = scf.if %is_active -> (f32, f32) {
//                     scf.yield %d_f32, %f1 : f32, f32
//                 } else {
//                     %old_dist = tensor.extract %d_acc[%j] : tensor<?xf32>
//                     %old_vis = tensor.extract %vis_acc[%j] : tensor<?xf32>
//                     scf.yield %old_dist, %old_vis : f32, f32
//                 }
                
//                 %v_inserted = tensor.insert %val into %v_acc[%j] : tensor<?xf32>
//                 %vis_inserted = tensor.insert %new_vis into %vis_acc[%j] : tensor<?xf32>
//                 %dist_inserted = tensor.insert %new_dist into %d_acc[%j] : tensor<?xf32>
                
//                 scf.yield %v_inserted, %vis_inserted, %dist_inserted : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
//             }
//             scf.yield %v_next, %visited_next, %dist_next : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
//         }

//         // --- 4. Write back to C++ MemRef ---
//         // Extract the fully populated distance tracker and write it back to memory
//         scf.for %k = %c0 to %num_nodes step %c1 {
//             %d_val = tensor.extract %final_results#2[%k] : tensor<?xf32>
//             memref.store %d_val, %out_dist[%k] : memref<?xf32>
//         }

//         return
//     }
// }








#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
    func.func private @start_timer()
    func.func private @stop_timer()

    func.func @bfs(%nodes_i32: i32, %edges_i32: i32, %coords_mem: memref<?x2xi64>, %vals_mem: memref<?xf32>, %out_dist: memref<?xf32>) attributes { llvm.emit_c_interface } {
        
        %num_nodes = arith.index_cast %nodes_i32 : i32 to index
        %num_edges = arith.index_cast %edges_i32 : i32 to index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32
        %f_neg1 = arith.constant -1.0 : f32

        // 1. Load Coordinates from MemRef
        %empty_coords = tensor.empty(%num_edges) : tensor<?x2xi64>
        %coords_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_coords = %empty_coords) -> tensor<?x2xi64> {
            %row = memref.load %coords_mem[%e, %c0] : memref<?x2xi64>
            %col = memref.load %coords_mem[%e, %c1] : memref<?x2xi64>
            %t1 = tensor.insert %row into %t_coords[%e, %c0] : tensor<?x2xi64>
            %t2 = tensor.insert %col into %t1[%e, %c1] : tensor<?x2xi64>
            scf.yield %t2 : tensor<?x2xi64>
        }

        // 2. Load Values from MemRef
        %empty_vals = tensor.empty(%num_edges) : tensor<?xf32>
        %vals_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_vals = %empty_vals) -> tensor<?xf32> {
            %v = memref.load %vals_mem[%e] : memref<?xf32>
            %t1 = tensor.insert %v into %t_vals[%e] : tensor<?xf32>
            scf.yield %t1 : tensor<?xf32>
        }

        // 3. Generate #CSR using dynamic arguments!
        %A = gblas.from_coo %coords_tensor, %vals_tensor (%num_nodes, %num_nodes)
            : tensor<?x2xi64>, tensor<?xf32> -> tensor<?x?xf32, #CSR>

        // 4. Initial Vectors
        %empty_f32 = tensor.empty(%num_nodes) : tensor<?xf32>
        
        %v_start_0 = linalg.fill ins(%f0 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
        %v_start = tensor.insert %f1 into %v_start_0[%c0] : tensor<?xf32>
        
        %visited_start_0 = linalg.fill ins(%f0 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
        %visited_start = tensor.insert %f1 into %visited_start_0[%c0] : tensor<?xf32>
        
        %dist_start_neg = linalg.fill ins(%f_neg1 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
        %dist_start = tensor.insert %f0 into %dist_start_neg[%c0] : tensor<?xf32>

        func.call @start_timer() : () -> () //time start

        // 5. BFS Loop
        %final_results:3 = scf.for %i = %c0 to %num_nodes step %c1 
            iter_args(%v_curr = %v_start, %visited_curr = %visited_start, %dist_curr = %dist_start) 
            -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
            
            %scratch_empty = tensor.empty(%num_nodes) : tensor<?xf32>
            %zeroed_scratch = linalg.fill ins(%f0 : f32) outs(%scratch_empty : tensor<?xf32>) -> tensor<?xf32>

            %v_computed = gblas.vxm %v_curr, %A outs(%zeroed_scratch), %visited_curr
                combine = multiplies reduce = plus {mask_complement = true}
                : tensor<?xf32>, tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32> 
                -> tensor<?xf32>

            %i_i32 = arith.index_cast %i : index to i32
            %i_f32 = arith.sitofp %i_i32 : i32 to f32
            %d_f32 = arith.addf %i_f32, %f1 : f32

            %v_next, %visited_next, %dist_next = scf.for %j = %c0 to %num_nodes step %c1 
                iter_args(%v_acc = %v_curr, %vis_acc = %visited_curr, %d_acc = %dist_curr) 
                -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
                
                %val = tensor.extract %v_computed[%j] : tensor<?xf32>
                %is_active = arith.cmpf ogt, %val, %f0 : f32
                
                %new_dist, %new_vis = scf.if %is_active -> (f32, f32) {
                    scf.yield %d_f32, %f1 : f32, f32
                } else {
                    %old_dist = tensor.extract %d_acc[%j] : tensor<?xf32>
                    %old_vis = tensor.extract %vis_acc[%j] : tensor<?xf32>
                    scf.yield %old_dist, %old_vis : f32, f32
                }
                
                %v_inserted = tensor.insert %val into %v_acc[%j] : tensor<?xf32>
                %vis_inserted = tensor.insert %new_vis into %vis_acc[%j] : tensor<?xf32>
                %dist_inserted = tensor.insert %new_dist into %d_acc[%j] : tensor<?xf32>
                
                scf.yield %v_inserted, %vis_inserted, %dist_inserted : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
            }
            scf.yield %v_next, %visited_next, %dist_next : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
        }

        func.call @stop_timer() : () -> () //time end

        // 6. Write back to C++
        scf.for %k = %c0 to %num_nodes step %c1 {
            %d_val = tensor.extract %final_results#2[%k] : tensor<?xf32>
            memref.store %d_val, %out_dist[%k] : memref<?xf32>
        }

        return
    }
}